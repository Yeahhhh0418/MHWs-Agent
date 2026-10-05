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

logger = get_logger("rag.base")  # [TODO-TECHDEBT] 日志初始化方式待确认


def load_jsonl(file_path: str) -> list:
    """读取 JSON Lines 文件，返回 list[dict]；空行跳过。"""
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
    _ID_FIELDS = ()  #子类声明用哪几个字段拼编号
    FIELD_LABELS: ClassVar[dict] = {}   # 子类声明：字段名 -> 中文标签，拼正文时用
    VALUE_LABELS: ClassVar[dict] = {}   # 子类可选：{字段名: {原值: 中文}}，如 parts_type 的 HELM -> 头盔

    def __init__(self):
        """从配置读路径与检索参数；子类靠 _DB_NAME 拿到自己那一套。"""
        cfg = get_config()
        r = cfg["retrieval"][self._DB_NAME]

        self._data_path = cfg["paths"][f"{self._DB_NAME}_data"]
        self._persist_dir = cfg["paths"][f"{self._DB_NAME}_persist"]

        self._embedding_model = cfg["embedding"]["model"]
        self._device = cfg["embedding"].get("device", "cpu")
        self._normalize = cfg["embedding"].get("normalize", True)

        self._bm25_k = r["bm25_k"]
        self._vector_k = r["vector_k"]
        self._bm25_weight = r["bm25_weight"]
        self._vector_weight = r["vector_weight"]
        self._rrf_c = r["rrf_c"]                      # RRF 平滑常数（不是返回条数）
        self._ensemble_top_n = r["ensemble_top_n"]    # 融合后真正保留的条数
        self._vector_score_threshold = r["vector_score_threshold"]

        self.is_ready = False

        self._embeddings = None

        self._docs = []
        self._vectorstore = None
        self._retriever = None

    def setup(self) -> "BaseRAG":
        """首次使用前调用一次；顺序不能换：先有数据才能编号，先有库才能建检索器。"""
        load_dotenv()
        self._embeddings = HuggingFaceEmbeddings(
            model_name=self._embedding_model,
            model_kwargs={"device": self._device},
            encode_kwargs={"normalize_embeddings": self._normalize},
        )
        self._load_data()      # ← 基类默认读 jsonl；怪物走 md，自己覆写
        self._assign_doc_ids()     # ← 给每条 Document 编个货号
        self._init_vectorstore()   # ← 基类写好
        self._init_retriever()     # ← 基类写好（混合检索）
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
        if not self.FIELD_LABELS:
            raise ValueError(f"{self.__class__.__name__} 没声明 FIELD_LABELS，正文会漏出英文 key")

        logger.info("加载数据", extra={"db": self._DB_NAME, "path": self._data_path})
        documents = []
        for data in load_jsonl(self._data_path):
            parts = []
            for k, v in data.items():
                value_map = self.VALUE_LABELS.get(k)
                if value_map and v in value_map:
                    v = value_map[v]

                label = self.FIELD_LABELS.get(k, k)
                if isinstance(v, list):
                    value_str = "、".join(str(item) for item in v)
                else:
                    value_str = str(v)
                parts.append(f"{label}：{value_str}")

            documents.append(Document(
                page_content="，".join(parts),
                metadata=data,
            ))

        self._docs = documents
        logger.info("数据加载完成", extra={"db": self._DB_NAME, "count": len(documents)})

    def _assign_doc_ids(self) -> None:
        """
        [WHY] 货号同时被两处依赖：EnsembleRetriever 靠它去重、评测靠它核对命中，
              而最早拿行号当编号，数据一重排就全变，评测会静默失效。
        [FIX] 编号只依赖业务字段 —— 字段不变，编号就不变。
        """
        for doc in self._docs:
            parts = []
            for field in self._ID_FIELDS:
                parts.append(str(doc.metadata[field]))
            doc.metadata[self._ID_KEY] = f"{self._DB_NAME}|" + "|".join(parts)

    def _init_vectorstore(self) -> None:
        """
        [WHY] 判断「建过库没有」只看目录里有没有文件，不看数据变没变 —— 改了数据不删库，
              就会出现 BM25 用新数据、向量路用旧数据的混装结果（曾见一半「头盔」一半「HELM」）。
        [FIX] 这里只管加载或首次构建；改了 _load_data 或数据文件后，必须手动删掉
              storage/ 下对应目录再跑。
        """
        if os.path.exists(self._persist_dir) and os.listdir(self._persist_dir):
            logger.info(f"检测到已有{self._DB_NAME}向量库，直接加载", extra={"db": self._DB_NAME})
            self._vectorstore = Chroma(
                embedding_function=self._embeddings,
                persist_directory=self._persist_dir,
            )
        else:
            logger.info(f"正在构建{self._DB_NAME}向量库（首次较慢）", extra={"db": self._DB_NAME})
            self._vectorstore = Chroma.from_documents(
                documents=self._docs,
                embedding=self._embeddings,
                persist_directory=self._persist_dir,
                collection_configuration={"hnsw": {"space": "cosine"}}
            )

    def _init_retriever(self) -> None:
        """
        [WHY] 单靠向量对专有名词不敏感，单靠 BM25 在中文上又不可靠（默认按空格分词）。
        [REF] ⚠️ 字符级分词分不开「火龙/雌火龙/护火龙」，
              monster 库已把 bm25_weight 降为 0（config.yaml）
        """
        bm25_retriever = BM25Retriever.from_documents(
            self._docs, preprocess_func=chinese_char_tokenizer
        )
        bm25_retriever.k = self._bm25_k

        vector_retriever = self._vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": self._vector_k,
                "score_threshold": self._vector_score_threshold,
            }
        )

        self._retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
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
        if not self.is_ready:
            raise RuntimeError(f"{self.__class__.__name__} 尚未初始化，请先调用 .setup()")
        return self._top_n(self._retriever.invoke(question))

