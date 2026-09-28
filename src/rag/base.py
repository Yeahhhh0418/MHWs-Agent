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
import os as _os  # [TODO-TECHDEBT] 别名无实际作用，可直接 import os

logger = get_logger("rag.base")  # [TODO-TECHDEBT] 日志初始化方式待确认

class BaseRAG(ABC):
    """
    [WHY] 四个库（武器/防具/素材/怪物）除「数据怎么读」外流程完全相同，抽成基类免得写四遍。
    [FIX] 子类只实现 _load_data，其余按 _DB_NAME 从配置取参数后统一走完。
    [REF] 历史遗留-待确认
    """

    # 子类需要覆写的属性:
    _DB_NAME = ""
    RAG_PROMPT_TEMPLATE = ""   # 各自的提示词模板

    # 内部主键名（去重用的货号）
    _ID_KEY = "_doc_id"
    _ID_FIELDS = ()  #子类声明用哪几个字段拼编号

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

        self._llm = None
        self._embeddings = None

        self._docs = []
        self._vectorstore = None
        self._retriever = None
        self._chain = None

    def setup(self) -> "BaseRAG":
        """首次使用前调用一次；顺序不能换：先有数据才能编号，先有库才能建检索器。"""
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
        """子类实现：把数据文件（JSON 或 Markdown）读成 self._docs；四个库的差异只在这里。"""
        pass

    def _assign_doc_ids(self) -> None:
        """
        [WHY] 货号同时被两处依赖：EnsembleRetriever 靠它去重、评测靠它核对命中，
              而最早拿行号当编号，数据一重排就全变，评测会静默失效。
        [FIX] 改用子类声明的业务字段拼（monster|火龙|狩猎提示），业务字段不变编号就不变。
        [REF] 怪物库改造指南.md 第三节「编号怎么定」
        """
        for doc in self._docs:
            parts = []
            for field in self._ID_FIELDS:
                parts.append(str(doc.metadata[field]))
            doc.metadata[self._ID_KEY] = f"{self._DB_NAME}|" + "|".join(parts)

    def _init_vectorstore(self):
        """
        [WHY] 判断「建过库没有」只看目录里有没有文件，不看数据变没变 —— 改了数据不删库，
              就会出现 BM25 用新数据、向量路用旧数据的混装结果（曾见一半「头盔」一半「HELM」）。
        [FIX] 这里只管加载或首次构建；改了 _load_data 或数据文件后，必须手动删掉
              storage/ 下对应目录再跑。
        [REF] 历史遗留-待确认
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
            """
            [WHY] 单靠向量对专有名词不敏感，单靠 BM25 在中文上又不可靠（默认按空格分词）。
            [FIX] 两路并用、按排名用 RRF 融合；中文那路配自写的字符级分词器。
            [REF] rag/tokenizer.py。⚠️ 字符级分词分不开「火龙/雌火龙/护火龙」，
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
        [FIX] 融合后统一截断到 ensemble_top_n，改条数只动配置。
        [REF] 历史遗留-待确认
        """
        return docs[: self._ensemble_top_n]

    def _init_chain(self):
        """组装 RAG 链：检索 → 截断 → 拼上下文 → 提示词 → LLM；模板需 {context} 与 {question} 两个坑位。"""
        rag_prompt = ChatPromptTemplate.from_template(self.RAG_PROMPT_TEMPLATE)
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
        """检索加生成，返回答案文本（str）；未 setup 直接报错，免得拿到半成品结果。"""
        if not self.is_ready:
            raise RuntimeError(f"{self.__class__.__name__} 尚未初始化，请先调用 .setup()")
        return self._chain.invoke(question)

    def __call__(self, question: str) -> str:
        """让实例能当函数直接调用，等价于 query(question)。"""
        return self.query(question)

    def retrieve(self, question: str) -> list:
        """
        [WHY] 检索节点只要文档和编号、不要生成（生成放在图的最后一步），
              评测也靠这个入口单独核对命中。
        [FIX] 只走到检索器，返回截断后的 Document 列表。
        [REF] src/agent/nodes.py 的 retrieve_node；src/eval/run_eval.py
        """
        if not self.is_ready:
            raise RuntimeError(f"{self.__class__.__name__} 尚未初始化，请先调用 .setup()")
        return self._top_n(self._retriever.invoke(question))

    @staticmethod
    def _format_docs(docs) -> str:
        """把多条 Document 的正文拼成一个上下文串（str），空行分隔。"""
        return "\n\n".join(doc.page_content for doc in docs)
