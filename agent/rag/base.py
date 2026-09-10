"""
agent/rag/base.py —— 所有 RAG 引擎的抽象基类
三个 RAG 库共享的流程（加载数据→建向量库→建检索器→组链）都写在这里，
子类只需要提供：自己的数据加载逻辑(_load_data) 和 自己的提示词模板(RAG_PROMPT_TEMPLATE)。
"""

import os as _os
from abc import ABC, abstractmethod

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from agent.chatllm import llm
from agent.config import get_config
from agent.logger import get_logger
from agent.rag.tokenizer import chinese_char_tokenizer

logger = get_logger("rag.base")


class BaseRAG(ABC):
    # 子类必须覆写这两个类属性
    _DB_NAME = ""              # "equipment" / "item" / "monster"
    RAG_PROMPT_TEMPLATE = ""   # 各自的提示词模板

    # 子类可覆写：俗称→正式词的同义词扩展表（默认空，不扩展）。
    # 例：{"手": "腕甲 腕部"} 表示查询里出现"手"时，追加"腕甲 腕部"帮助命中。
    QUERY_SYNONYMS = {}

    def __init__(self):
        # 从 config.yaml 读取本库的配置。用 _DB_NAME 区分是哪个库。
        cfg = get_config()
        r = cfg["retrieval"][self._DB_NAME]

        self._data_path = cfg["paths"][f"{self._DB_NAME}_data"]
        self._persist_dir = cfg["paths"][f"{self._DB_NAME}_persist"]
        self._embedding_model = cfg["embedding"]["model"]
        self._bm25_k = r["bm25_k"]
        self._vector_k = r["vector_k"]
        self._bm25_weight = r["bm25_weight"]
        self._vector_weight = r["vector_weight"]
        self._ensemble_n_results = r["ensemble_n_results"]
        self._device = cfg["embedding"].get("device", "cpu")
        self._normalize = cfg["embedding"].get("normalize", True)

        self.is_ready = False
        self._chain = None
        self._retriever = None
        self._vectorstore = None
        self._docs = []
        self._llm = None
        self._embeddings = None


    # ── 模板方法：固定 5 步顺序 ───────────────────────────
    def setup(self) -> "BaseRAG":
        load_dotenv()
        self._llm = llm
        self._embeddings = HuggingFaceEmbeddings(
            model_name=self._embedding_model,
            model_kwargs={"device": self._device},
            encode_kwargs={"normalize_embeddings": self._normalize},
        )
        try:
            self._load_data()      # ← 子类填（抽象方法）
        except Exception as e:
            logger.error("数据加载失败，置为空列表", extra={"db": self._DB_NAME, "error": str(e)})
            self._docs = []
        self._init_vectorstore()   # ← 基类写好
        self._init_retriever()     # ← 基类写好（混合检索）
        self._init_chain()         # ← 基类写好
        self.is_ready = True
        return self

    # ── 子类必须实现的唯一方法 ────────────────────────────
    @abstractmethod
    def _load_data(self):
        """加载 JSON 并填充 self._docs。子类必须实现。"""

    # ── 基类共享实现 ──────────────────────────────────────
    def _init_vectorstore(self):
        """构建或加载 Chroma 向量库（三个库逻辑完全一样）。"""
        if _os.path.exists(self._persist_dir) and _os.listdir(self._persist_dir):
            logger.info("检测到已有向量库，直接加载", extra={"db": self._DB_NAME})
            self._vectorstore = Chroma(
                embedding_function=self._embeddings,
                persist_directory=self._persist_dir,
            )
        else:
            logger.info("正在构建向量库（首次较慢）", extra={"db": self._DB_NAME})
            self._vectorstore = Chroma.from_documents(
                documents=self._docs,
                embedding=self._embeddings,
                persist_directory=self._persist_dir,
            )

    def _init_retriever(self):
        """混合检索器：BM25（关键词）+ Chroma（向量），用 RRF 合并。三个库统一。"""
        bm25_retriever = BM25Retriever.from_documents(
            self._docs, preprocess_func=chinese_char_tokenizer
        )
        bm25_retriever.k = self._bm25_k

        vector_retriever = self._vectorstore.as_retriever(
            search_kwargs={"k": self._vector_k}
        )

        self._retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[self._bm25_weight, self._vector_weight],
            c=self._ensemble_n_results,
        )

    def _init_chain(self):
        """组装 RAG 链：检索 → 格式化 → 提示词 → LLM → 输出。"""
        rag_prompt = ChatPromptTemplate.from_template(self.RAG_PROMPT_TEMPLATE)
        self._chain = (
            {
                "context": self._retriever | self._format_docs,
                "question": RunnablePassthrough(),
            }
            | rag_prompt
            | self._llm
            | StrOutputParser()
        )

    # ── 查询接口 ──────────────────────────────────────────
    def _expand_query(self, question: str) -> str:
        """查询前做同义词扩展：把俗称部位词（如"手"）补上正式词（如"腕甲 腕部"）。"""
        for collo, formal in self.QUERY_SYNONYMS.items():
            if collo in question:
                question = f"{question} {formal}"
        return question

    def query(self, question: str) -> str:
        if not self.is_ready:
            raise RuntimeError(f"{self.__class__.__name__} 尚未初始化，请先调用 .setup()")
        question = self._expand_query(question)
        return self._chain.invoke(question)

    def __call__(self, question: str) -> str:
        return self.query(question)

    def retrieve(self, question: str) -> list:
        """只检索、不生成。返回命中的 Document 列表（Phase 3 调参用）。"""
        if not self.is_ready:
            raise RuntimeError(f"{self.__class__.__name__} 尚未初始化，请先调用 .setup()")
        question = self._expand_query(question)
        return self._retriever.invoke(question)

    @staticmethod
    def _format_docs(docs) -> str:
        """把多个 Document 拼成一个上下文字符串。"""
        return "\n\n".join(doc.page_content for doc in docs)