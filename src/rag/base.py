from abc import ABC, abstractmethod
from mhwi_agent.config import get_config
from models.chatllm import get_llm
from mhwi_agent.logger import get_logger
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from rag.tokenizer import chinese_char_tokenizer
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os as _os  #这里什么要这样命名？

logger = get_logger("rag.base")  #暂时还不清楚日志创建

class BaseRAG(ABC):
    # 子类需要覆写的属性:
    _DB_NAME = ""
    RAG_PROMPT_TEMPLATE = ""   # 各自的提示词模板

    # 内部主键名（去重用的货号）
    _ID_KEY = "_doc_id"
    _ID_FIELDS = ()  #子类声明用哪几个字段拼编号
    def __init__(self):
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

        self._llm = None
        self._embeddings = None

        self._docs = []
        self._vectorstore = None
        self._retriever = None
        self._chain = None

    def setup(self) -> "BaseRAG":
        load_dotenv()
        self._llm = get_llm()
        self._embeddings = HuggingFaceEmbeddings(
            model_name=self._embedding_model,
            model_kwargs={"device": self._device},
            encode_kwargs={"normalize_embeddings": self._normalize},
        )
        self._load_data()      # ← 子类填（抽象方法）
        self._assign_doc_ids()     # ← 给每条 Document 编个货号
        self._init_vectorstore()   # ← 基类写好
        self._init_retriever()     # ← 基类写好（混合检索）
        self._init_chain()         # ← 基类写好
        self.is_ready = True
        return self

    @abstractmethod
    def _load_data(self):
        """加载 JSON 并填充 self._docs。子类必须实现。"""
        pass

    def _assign_doc_ids(self) -> None:
        """给每条 Document 编一个货号 _doc_id。"""
        for doc in self._docs:
            parts = [str(doc.metadata[f]) for f in self._ID_FIELDS]
            doc.metadata[self._ID_KEY] = f"{self._DB_NAME}|" + "|".join(parts)

    def _init_vectorstore(self):
        """构建或加载 Chroma 向量库。

        注意：判断依据是「目录里有没有文件」，不是「数据有没有变」。
        所以你改了 _load_data 或数据文件之后，必须手动把 storage/ 下对应的
        库目录删掉再跑，否则向量库不会更新，BM25 用新数据、向量路还是旧数据，
        检索结果会新旧混在一起（比如一半显示「头盔」、一半显示「HELM」）。
        """
        if _os.path.exists(self._persist_dir) and _os.listdir(self._persist_dir):
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

    def _init_retriever(self):
            """混合检索器：BM25（关键词）+ Chroma（向量），用 RRF 合并。三个库统一。
                            RRF 是一种合并多路排名的算法"""
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
        """融合后截断，避免把全部召回都塞进 LLM 上下文。"""
        return docs[: self._ensemble_top_n]

    def _init_chain(self):
        """组装 RAG 链：检索 → 截断 → 格式化 → 提示词 → LLM → 输出。"""
        rag_prompt = ChatPromptTemplate.from_template(self.RAG_PROMPT_TEMPLATE)#RAG_PROMPT_TEMPLATE有两个"坑位"：{context} 和 {question}，后面会把检索到的内容和用户问题填进去。
        self._chain = (
            {
                "context": self._retriever | self._top_n | self._format_docs,
                "question": RunnablePassthrough(),
            }
            | rag_prompt
            | self._llm
            | StrOutputParser()
        )

    # ── 查询接口 ──────────────────────────────────────────
    def query(self, question: str) -> str:
        if not self.is_ready:
            raise RuntimeError(f"{self.__class__.__name__} 尚未初始化，请先调用 .setup()")
        return self._chain.invoke(question)
    
    def __call__(self, question: str) -> str:
        return self.query(question)
    
    def retrieve(self, question: str) -> list:
        """只检索、不生成。返回命中的 Document 列表（已按 ensemble_top_n 截断）。"""
        if not self.is_ready:
            raise RuntimeError(f"{self.__class__.__name__} 尚未初始化，请先调用 .setup()")
        return self._top_n(self._retriever.invoke(question))

    @staticmethod
    def _format_docs(docs) -> str:
        """把多个 Document 拼成一个上下文字符串。"""
        return "\n\n".join(doc.page_content for doc in docs)