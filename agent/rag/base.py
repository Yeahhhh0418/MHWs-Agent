class BaseRAG(ABC):
    """所有 RAG 引擎的抽象基类。
    
    setup() 定义初始化管线（不变）：
        _load_data() → _init_vectorstore() → _init_retriever() → _init_chain()
    
    子类只需覆写：
        _load_data()       — JSON 数据加载 + Document 构建
        _init_retriever()  — 检索器创建（纯向量 / 混合）
    """

    def __init__(self, data_path, persist_dir, embedding_model,
                 bm25_k, vector_k, bm25_weight, vector_weight, ensemble_n_results):
        # 所有参数从 config 显式传入，无隐式全局变量
        ...

    def setup(self) -> "BaseRAG":
        load_dotenv()
        self._llm = llm
        self._embeddings = HuggingFaceEmbeddings(...)
        self._load_data()
        self._init_vectorstore()      # 共享实现
        self._init_retriever()        # 子类覆写
        self._init_chain()            # 共享实现
        self.is_ready = True
        return self

    @abstractmethod
    def _load_data(self): ...

    @abstractmethod
    def _init_retriever(self): ...

    def _init_vectorstore(self):
        """ChromaDB 构建/加载，所有子类逻辑一致"""
        if _os.path.exists(self._persist_dir) and _os.listdir(self._persist_dir):
            self._vectorstore = Chroma(
                embedding_function=self._embeddings,
                persist_directory=self._persist_dir,
            )
        else:
            self._vectorstore = Chroma.from_documents(
                documents=self._docs,
                embedding=self._embeddings,
                persist_directory=self._persist_dir,
            )
        

    def _init_chain(self):
        """LCEL 链组装，所有子类逻辑一致"""
        
    def query(self, question: str) -> str:
        if not self.is_ready:
            raise RuntimeError(...)
        return self._chain.invoke(question)