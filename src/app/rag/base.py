import hashlib
import json
import os
import shutil
from abc import ABC
from pathlib import Path
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

# 数据指纹存在库目录里，文件名固定。
# ⚠️ 算「建过库没有」时要把它排除掉 —— 不然一个只有指纹、没有别的文件的空目录，
#    会被当成"库已经建好了"。
_FINGERPRINT_FILE = "_data_fingerprint.txt"


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

    def _data_fingerprint(self) -> str:
        """
        [WHY] 判断"库还能不能用"要看**要入库的内容**变没变 —— 只看数据文件不够，
              改了 _load_data（拼正文的规则）也一样会让库作废。
        [FIX] 把每条文档的「编号 + 正文」序列化后算 sha256，取前 16 位。
              顺序也进指纹：宁可敏感（多重建一次），也别漏判（旧数据静默生效）。
        """
        payload = []
        for doc in self._docs:
            payload.append({"id": doc.metadata[self._ID_KEY], "content": doc.page_content})
        blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def _init_vectorstore(self) -> None:
        """
        [WHY] 判断「建过库没有」原来只看目录里有没有文件 —— 改了数据不删库，就会出现
              BM25 用新数据、向量路用旧数据的混装结果（曾见一半「头盔」一半「HELM」）。
        [FIX] 把要入库的内容算个指纹存进库目录，加载前比一比：对得上就加载，
              对不上就删掉重建。不用再让人记着手动删。
        """
        fp_path = Path(self._persist_dir) / _FINGERPRINT_FILE
        current = self._data_fingerprint()

        # 「建过库没有」= 目录里有东西，而且不只有那个指纹文件
        has_store = os.path.exists(self._persist_dir) and any(
            name != _FINGERPRINT_FILE for name in os.listdir(self._persist_dir)
        )
        stored = fp_path.read_text(encoding="utf-8").strip() if fp_path.exists() else None

        if has_store and stored == current:
            logger.info(
                f"检测到已有{self._DB_NAME}向量库，数据指纹一致，直接加载",
                extra={"db": self._DB_NAME, "fingerprint": current},
            )
            self._vectorstore = Chroma(
                embedding_function=self._embeddings,
                persist_directory=self._persist_dir,
            )
            return

        if has_store:
            # ⚠️ 重建前必须先删干净：Chroma 的 from_documents 是**追加**不是覆盖，
            #    不删的话旧的还留在库里，结果比不重建更糟（实测 3 条 + 5 条 = 8 条）
            logger.warning(
                f"{self._DB_NAME}向量库跟当前数据对不上，重建",
                extra={"db": self._DB_NAME, "stored": stored, "current": current},
            )
            try:
                shutil.rmtree(self._persist_dir)
            except OSError as e:
                # Windows 上如果**同一个进程里已经打开过**这个库，chroma.sqlite3 还占着，
                # 删不掉。正常用法碰不到（每个库在进程里只建一次），脚本/notebook 里可能。
                # 这里宁可响亮地报错，也别退回"不删直接建" —— 实测那样会追加成两倍条数。
                raise RuntimeError(
                    f"{self._DB_NAME} 向量库要重建，但删不掉旧目录 {self._persist_dir} —— "
                    f"它还被占着（Windows 上同一个进程里二次打开就会这样）。"
                    f"换个新进程跑，或者手动删掉这个目录再跑。"
                ) from e
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
        # 建完把这次的指纹记下来，下次加载前拿它比
        fp_path.parent.mkdir(parents=True, exist_ok=True)
        fp_path.write_text(current, encoding="utf-8")
        logger.info(
            f"{self._DB_NAME}向量库构建完成",
            extra={"db": self._DB_NAME, "fingerprint": current},
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
