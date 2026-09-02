"""
怪物猎人荒野 - 物品图鉴 RAG 模块
===============================
提供 ItemDataRAG 类，封装物品数据的检索增强生成（Hybrid RAG）。
支持在节点函数中直接实例化和调用。

用法示例:
    from ItemData_rag import ItemDataRAG

    item_rag = ItemDataRAG()
    item_rag.setup()                      # 首次运行构建数据库，之后自动加载
    answer = item_rag.query("回复药是做什么的？")
    print(answer)
"""

import json
import os as _os

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from agent.chatllm import llm

# ── 常量 ─────────────────────────────────────────────────────
# 物品类型中文映射
TYPE_CN_MAP = {
    "EXPENDABLE": "消耗品",
    "TOOL": "道具/工具",
    "MATERIAL": "素材/材料",
    "SHELL": "弹药/瓶",
    "GEM": "装饰品/宝珠",
    "POINT": "特产/点数物品",
    "BOTTLE": "瓶",
}

# RAG 提示词模板
RAG_PROMPT_TEMPLATE = """你是一个专业的《怪物猎人：荒野》物品图鉴助手。请根据以下检索到的物品信息，回答玩家的问题。

【核心规则——务必遵守】：
玩家经常使用不精确的俗称或简称来描述物品，你必须根据检索结果自动匹配最接近的正式物品名。例如：

| 玩家可能的说法 | 实际指的游戏物品 | 推理逻辑 |
|--------------|----------------|--------|
| "回复药" | 【回复药】 | 直接匹配正式名称 |
| "大回复药" / "高级回复药" | 【回复药G】 | "大/高级"对应 G 级升级版 |
| "秘药" | 【秘药】 | 直接匹配 |
| "远古药" / "古代秘药" | 【远古秘药】 | "远古/古代"对应远古前缀 |
| "解毒的" / "解毒东西" | 【解毒药】 | 功能描述匹配 |
| "烤肉" / "烤熟的" | 【烤熟的肉】 | 俗称匹配 |
| "粉尘" / "生命粉" | 【生命粉尘】 | 简称匹配 |
| "陷阱" / "麻痹陷阱" | 【麻痹陷阱】 | 功能+类型匹配 |
| "捕获球" / "麻醉球" | 【捕获用麻醉球】 | 功能描述匹配 |
| "闪光弹" / "闪光" | 【闪光弹】 | 简称匹配 |
| "怪物的皮" / "龙皮" | 各种怪物皮素材 | 模糊描述匹配素材类 |

【物品类型说明】：
- 消耗品(EXPENDABLE)：回复药、肉、粉尘、果实等可使用的物品
- 道具/工具(TOOL)：陷阱、爆弹、投射器弹药等功能性道具
- 素材/材料(MATERIAL)：从怪物身上剥取的皮、鳞、爪等制作材料
- 弹药/瓶(SHELL)：弓弩用的各种弹药和瓶
- 装饰品/宝珠(GEM)：镶嵌在装备上的宝珠
- 特产/点数物品(POINT)：可用于交换点数的特产
- 瓶(BOTTLE)：可装备在弓上的瓶

【回答规范】：
1. 绝对不要说"没有找到你说的XX"或"攻略信息中没有XX"——只要检索结果里有名称或功能相似的物品，就直接告诉玩家。
2. 回答时先用一句话确认匹配，例如"你说的应该是【回复药G】"，然后给出物品详情。
3. 如果检索结果中有多个可能匹配的物品（如回复药、回复药G、远古秘药），把它们都列出来并说明区别（稀有度、效果、价格等）。
4. 回答应包含：物品名、类型、稀有度、效果/说明、售价/购买价。
5. 如果玩家问的是"做什么用的"或"怎么获得"，根据物品说明给出解释。
6. 只有在检索结果中确实完全没有任何相关物品时，才回答"抱歉，我的物品图鉴里暂时没有找到这个信息"。

【检索到的物品信息】：
{context}

【玩家的问题】：
{question}

【你的回答】："""


# ── 中文字符级分词器 ─────────────────────────────────────────
def chinese_char_tokenizer(text: str) -> list[str]:
    """中文字符级分词器，供 BM25 关键词检索使用。

    BM25 默认用空格分词，对中文无效。此分词器将每个汉字拆成独立 token，
    英文/数字保持连续，确保"锁刃"这类关键词检索不会遗漏。
    """
    tokens = []
    buf = ""
    for ch in text:
        if "一" <= ch <= "鿿":  # CJK 汉字
            if buf:
                tokens.append(buf)
                buf = ""
            tokens.append(ch)
        elif ch.isalnum():
            buf += ch
        else:
            if buf:
                tokens.append(buf)
                buf = ""
    if buf:
        tokens.append(buf)
    return [t for t in tokens if t]


# ── 主类 ─────────────────────────────────────────────────────
class ItemDataRAG:
    """物品图鉴 RAG 引擎。

    封装了物品数据的加载、向量数据库构建、混合检索（BM25 + 向量）、
    以及 LLM 生成回答的完整流程。可在节点函数中直接使用。

    属性:
        is_ready (bool): 是否已完成初始化。

    用法:
        rag = ItemDataRAG()
        rag.setup()
        answer = rag.query("回复药是做什么的？")
        # 或作为节点函数
        def item_node(state):
            return {"answer": rag.query(state["question"])}
    """

    def __init__(
        self,
        data_path: str = "data/ItemData.json",
        persist_dir: str = "./mh_item_chroma_db",
        embedding_model: str = "BAAI/bge-small-zh-v1.5",
        bm25_k: int = 8,
        vector_k: int = 8,
    ):
        """初始化 ItemDataRAG。

        参数:
            data_path: 物品 JSON 数据文件路径。
            persist_dir: Chroma 向量数据库持久化目录。
            embedding_model: HuggingFace embedding 模型名称。
            bm25_k: BM25 关键词检索返回的文档数。
            vector_k: 向量检索返回的文档数。
        """
        self._data_path = data_path
        self._persist_dir = persist_dir
        self._embedding_model = embedding_model
        self._bm25_k = bm25_k
        self._vector_k = vector_k

        self.is_ready = False
        self._chain = None
        self._retriever = None
        self._vectorstore = None
        self._docs = None
        self._llm = None

    # ── 属性 ──────────────────────────────────────────────

    @property
    def retriever(self):
        """获取混合检索器（BM25 + Chroma 向量）。"""
        return self._retriever

    @property
    def documents(self):
        """获取已加载的物品 Document 列表。"""
        return self._docs

    # ── 初始化 ────────────────────────────────────────────

    def setup(self) -> "ItemDataRAG":
        """执行完整初始化：加载数据、构建/加载向量库、组装链。

        首次运行会构建向量数据库并持久化；之后直接加载已有库。
        返回 self，支持链式调用。
        """
        load_dotenv()

        self._llm = llm
        self._embeddings = HuggingFaceEmbeddings(
            model_name=self._embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self._load_data()
        self._init_vectorstore()
        self._init_retriever()
        self._init_chain()
        self.is_ready = True
        return self

    def _load_data(self):
        """加载物品 JSON 数据，转为 LangChain Document 列表。"""
        print(f"正在加载怪猎物品种类: {self._data_path} ...")
        documents = []
        with open(self._data_path, "r", encoding="utf-8") as f:
            data_list = json.load(f)

        for item in data_list:
            name = item.get("name", "")
            if not name or name == "---":
                continue

            type_cn = TYPE_CN_MAP.get(item.get("type"), item.get("type", "未知"))
            rare = item.get("rare", "?")
            sell = item.get("sell_price", "?")
            buy = item.get("buy_price", "?")
            desc = item.get("description", "")

            content = (
                f"【{name}】是一类{type_cn}，稀有度 R{rare}。"
                f"说明：{desc}"
                f"售价 {sell}z / 购买价 {buy}z。"
            )

            metadata = {
                "item_id": item.get("item_id"),
                "name": name,
                "type": item.get("type"),
                "type_cn": type_cn,
                "rare": rare,
                "sell_price": sell,
                "buy_price": buy,
                "description": desc,
            }
            documents.append(Document(page_content=content, metadata=metadata))

        self._docs = documents
        print(f"成功加载 {len(documents)} 种物品数据！")

    def _init_vectorstore(self):
        """构建或加载 Chroma 向量数据库。"""
        if _os.path.exists(self._persist_dir) and _os.listdir(self._persist_dir):
            print("检测到已有物品向量数据库，直接加载...")
            self._vectorstore = Chroma(
                embedding_function=self._embeddings,
                persist_directory=self._persist_dir,
            )
        else:
            print("正在构建物品向量数据库（首次运行需要一点时间）...")
            self._vectorstore = Chroma.from_documents(
                documents=self._docs,
                embedding=self._embeddings,
                persist_directory=self._persist_dir,
            )
        print("物品向量数据库就绪！")

    def _init_retriever(self):
        """创建混合检索器：BM25（关键词） + Chroma（语义向量）。"""
        bm25_retriever = BM25Retriever.from_documents(
            self._docs, preprocess_func=chinese_char_tokenizer
        )
        bm25_retriever.k = self._bm25_k

        vector_retriever = self._vectorstore.as_retriever(
            search_kwargs={"k": self._vector_k}
        )

        self._retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.5, 0.5],
            c=60,
        )

    def _init_chain(self):
        """组装 RAG 链：检索 → 格式化 → 提示词 → LLM → 输出解析。"""
        rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

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

    def query(self, question: str) -> str:
        """根据问题检索物品信息并生成回答。

        参数:
            question: 玩家的问题（如 "回复药是做什么的？"）。

        返回:
            生成的回答字符串。

        异常:
            RuntimeError: 如果尚未调用 setup()。
        """
        if not self.is_ready:
            raise RuntimeError(
                "ItemDataRAG 尚未初始化，请先调用 .setup() 方法。"
            )
        return self._chain.invoke(question)

    def __call__(self, question: str) -> str:
        """快捷调用：实例可被当作函数直接调用。"""
        return self.query(question)

    # ── 工具方法 ──────────────────────────────────────────

    @staticmethod
    def _format_docs(docs: list[Document]) -> str:
        """将检索到的多个文档拼接为一个上下文字符串。"""
        return "\n\n".join(doc.page_content for doc in docs)


# ── 默认单例（可选） ──────────────────────────────────────────
_default_instance: ItemDataRAG | None = None


def get_item_rag() -> ItemDataRAG:
    """获取全局默认的 ItemDataRAG 单例（惰性初始化）。"""
    global _default_instance
    if _default_instance is None:
        _default_instance = ItemDataRAG()
        _default_instance.setup()
    return _default_instance


# ── 测试入口 ─────────────────────────────────────────────────
if __name__ == "__main__":
    rag = ItemDataRAG()
    rag.setup()

    print("\n--- 怪猎物品种类 Agent 已就绪 ---")
    while True:
        q = input("\n请输入你的问题 (输入 'quit' 退出): ")
        if q.lower() == "quit":
            break

        print("\n正在检索并生成回答...\n")
        response = rag.query(q)
        print(f"Agent: {response}")
