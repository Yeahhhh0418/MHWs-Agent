import json
import os
from abc import ABC
from typing import ClassVar

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import get_config
from app.logger import get_logger
from app.rag.tokenizer import chinese_char_tokenizer

logger = get_logger("rag.base")


def load_jsonl(file_path: str) -> list:
    """读取 JSON Lines 文件，返回 list[dict]；空行跳过。"""
    # 一行一条记录，空行跳过
    data_list = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data_list.append(json.loads(line))
    return data_list


class BaseRAG(ABC):
    """
    [WHY] 四个库（武器/防具/素材/怪物）除「数据怎么读」外流程完全相同，抽成基类免得写四遍。
    """

    # 子类需要覆写的属性:
    _DB_NAME = ""
    # 内部主键名（去重用的货号）
    _ID_KEY = "_doc_id"
    # 子类声明用哪几个字段拼编号。⚠️ 改了它、或改了数据里的标题，cases.jsonl 的
    # required_doc_ids 会静默失配 —— 召回率变 0 但不报错（见 CLAUDE.md 硬规矩 1）
    _ID_FIELDS = ()
    FIELD_LABELS: ClassVar[dict] = {}  # 子类声明：字段名 -> 中文标签，拼正文时用
    VALUE_LABELS: ClassVar[
        dict
    ] = {}  # 子类可选：{字段名: {原值: 中文}}，如 parts_type 的 HELM -> 头盔

    def __init__(self):
        """从配置读路径与检索参数；子类靠 _DB_NAME 拿到自己那一套。"""
        # 先定位自己那个库的配置块，后面都从这儿取
        cfg = get_config()
        r = cfg["retrieval"][self._DB_NAME]

        # 数据从哪儿读、向量库建到哪儿
        self._data_path = cfg["paths"][f"{self._DB_NAME}_data"]
        self._persist_dir = cfg["paths"][f"{self._DB_NAME}_persist"]

        # 向量模型与编码设置
        self._embedding_model = cfg["embedding"]["model"]
        self._device = cfg["embedding"].get("device", "cpu")
        self._normalize = cfg["embedding"].get("normalize", True)

        # 检索参数：两路各取几条、各占多少权重、融合后留几条
        self._bm25_k = r["bm25_k"]
        self._vector_k = r["vector_k"]
        self._bm25_weight = r["bm25_weight"]
        self._vector_weight = r["vector_weight"]
        self._rrf_c = r["rrf_c"]  # RRF 平滑常数（不是返回条数）
        self._ensemble_top_n = r["ensemble_top_n"]  # 融合后真正保留的条数
        self._vector_score_threshold = r["vector_score_threshold"]

        # 状态位：setup() 走完之前不许检索
        self.is_ready = False

        # 待 setup() 填充的运行时对象
        self._embeddings = None
        self._docs = []
        self._vectorstore = None
        self._retriever = None

    def setup(self) -> "BaseRAG":
        """首次使用前调用一次；顺序不能换：先有数据才能编号，先有库才能建检索器。"""
        # 先把 .env 读进来
        load_dotenv()
        # 建向量模型：首次要下载权重，之后就快了
        self._embeddings = HuggingFaceEmbeddings(
            model_name=self._embedding_model,
            model_kwargs={"device": self._device},
            encode_kwargs={"normalize_embeddings": self._normalize},
        )
        self._load_data()  # ← 基类默认读 jsonl；怪物走 md，自己覆写
        self._assign_doc_ids()  # ← 给每条 Document 编个货号
        self._init_vectorstore()  # ← 基类写好
        self._init_retriever()  # ← 基类写好（混合检索）
        self.is_ready = True
        return self

    def _load_data(self) -> None:
        """
        [WHY] 装备 / 武器 / 素材三个库原来是各自抄了一份 _load_data 的逻辑，
              唯一的差别只有字段标签和一处枚举翻译。
        [FIX] 默认实现读 jsonl，按 FIELD_LABELS 拼「标签：值」——列表用顿号连、字段之间用逗号连；
              VALUE_LABELS 可选，子类给了就把对应字段的值翻成中文。
              数据源不是 jsonl 的库（怪物走 md）整个覆写这个方法。
        """
        # 没声明标签就早点炸：否则正文里漏出英文 key，搜不到也读不懂
        if not self.FIELD_LABELS:
            raise ValueError(f"{self.__class__.__name__} 没声明 FIELD_LABELS，正文会漏出英文 key")

        logger.info("加载数据", extra={"db": self._DB_NAME, "path": self._data_path})
        # 一条数据拼成一个 Document
        documents = []
        for data in load_jsonl(self._data_path):
            # 逐字段拼成「中文标签：值」
            parts = []
            for k, v in data.items():
                # 子类给了映射就把枚举值翻成中文（HELM -> 头盔）
                value_map = self.VALUE_LABELS.get(k)
                if value_map and v in value_map:
                    v = value_map[v]

                # 没配标签的字段退回首字母，至少不丢内容
                label = self.FIELD_LABELS.get(k, k)
                # 列表用顿号连（一件装备的制作材料是一长串）
                if isinstance(v, list):
                    value_str = "、".join(str(item) for item in v)
                else:
                    value_str = str(v)
                parts.append(f"{label}：{value_str}")

            # metadata 存原值 —— 编号和去重都靠它，翻译只影响正文
            documents.append(
                Document(
                    page_content="，".join(parts),
                    metadata=data,
                )
            )

        self._docs = documents
        logger.info("数据加载完成", extra={"db": self._DB_NAME, "count": len(documents)})

    def _assign_doc_ids(self) -> None:
        """
        [WHY] 货号同时被两处依赖：EnsembleRetriever 靠它去重、评测靠它核对命中，
              而最早拿行号当编号，数据一重排就全变，评测会静默失效。
        [FIX] 编号只依赖业务字段 —— 字段不变，编号就不变。
        """
        for doc in self._docs:
            # 按 _ID_FIELDS 声明的顺序拼，顺序换了编号就换了
            parts = []
            for field in self._ID_FIELDS:
                parts.append(str(doc.metadata[field]))
            # 冠上库名，四个库的编号不会撞
            doc.metadata[self._ID_KEY] = f"{self._DB_NAME}|" + "|".join(parts)

    def _init_vectorstore(self) -> None:
        """
        [WHY] 判断「建过库没有」只看目录里有没有文件，不看数据变没变 —— 改了数据不删库，
              就会出现 BM25 用新数据、向量路用旧数据的混装结果（曾见一半「头盔」一半「HELM」）。
        [FIX] 这里只管加载或首次构建；改了 _load_data 或数据文件后，必须手动删掉
              storage/ 下对应目录再跑。
        """
        # 目录里有文件就直接加载 —— 内容对不对一概不管，所以改了数据必须手动删目录
        if os.path.exists(self._persist_dir) and os.listdir(self._persist_dir):
            logger.info(f"检测到已有{self._DB_NAME}向量库，直接加载", extra={"db": self._DB_NAME})
            self._vectorstore = Chroma(
                embedding_function=self._embeddings,
                persist_directory=self._persist_dir,
            )
        else:
            # 没有就现建，首次要等一会儿
            logger.info(f"正在构建{self._DB_NAME}向量库（首次较慢）", extra={"db": self._DB_NAME})
            # space=cosine 要跟 embeddings 那边的 normalize 配套
            self._vectorstore = Chroma.from_documents(
                documents=self._docs,
                embedding=self._embeddings,
                persist_directory=self._persist_dir,
                collection_configuration={"hnsw": {"space": "cosine"}},
            )

    def _init_retriever(self) -> None:
        """
        [WHY] 单靠向量对专有名词不敏感，单靠 BM25 在中文上又不可靠（默认按空格分词）。
        [REF] ⚠️ 字符级分词分不开「火龙/雌火龙/护火龙」，
              monster 库已把 bm25_weight 降为 0（config.yaml）
        """
        # 第一路：BM25 关键词召回。中文得换成字符级分词，默认按空格切等于没分
        bm25_retriever = BM25Retriever.from_documents(
            self._docs, preprocess_func=chinese_char_tokenizer
        )
        bm25_retriever.k = self._bm25_k

        # 第二路：向量召回。低于阈值的直接丢，所以一轮有可能返回 0 条
        vector_retriever = self._vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": self._vector_k,
                "score_threshold": self._vector_score_threshold,
            },
        )

        # 两路各自排名后用 RRF 融成一条榜单
        self._retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            # 权重配成 0 就等于关掉这一路（monster 关掉了 BM25）
            weights=[self._bm25_weight, self._vector_weight],
            c=self._rrf_c,
            # 按货号 _doc_id 去重：两路捞到同一条时合并成一条，
            # 并把两边的排名分数叠加起来（这会让「两路都命中」的条目排更前）
            id_key=self._ID_KEY,
        )

    def _top_n(self, docs: list) -> list:
        """
        [WHY] 召回条数不等于该喂给模型的条数，全塞进去既贵又吵。
        """
        return docs[: self._ensemble_top_n]

    # ── 查询接口 ──────────────────────────────────────────
    def retrieve(self, question: str) -> list:
        """
        [WHY] 检索节点只要文档和编号、不要生成（生成放在图的最后一步），
              评测也靠这个入口单独核对命中。
        [REF] src/app/eval/run_eval.py
        """
        # 没 setup 过就直接报错：不然会静默返回空结果，很难查
        if not self.is_ready:
            raise RuntimeError(f"{self.__class__.__name__} 尚未初始化，请先调用 .setup()")
        return self._top_n(self._retriever.invoke(question))
