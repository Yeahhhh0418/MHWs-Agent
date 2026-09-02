"""
怪物猎人荒野 - 怪物图鉴 RAG 模块
===============================
提供 MonsterDataRAG 类，封装怪物与生态数据的检索增强生成（Hybrid RAG）。
涵盖大型怪物（Monster）和环境生物（EnvironmentCreature）两类数据。
支持在节点函数中直接实例化和调用。

用法示例:
    from MonsterData_rag import MonsterDataRAG

    monster_rag = MonsterDataRAG()
    monster_rag.setup()                      # 首次运行构建数据库，之后自动加载
    answer = monster_rag.query("雌火龙怎么打？")
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
# 怪物种类中文映射（备用，大部分数据已有中文种类名）
CATEGORY_CN_MAP = {
    "飞龙种": "飞龙种",
    "鸟龙种": "鸟龙种",
    "牙兽种": "牙兽种",
    "鱼龙种": "鱼龙种",
    "兽龙种": "兽龙种",
    "海龙种": "海龙种",
    "古龙种": "古龙种",
    "造龙种": "造龙种",
    "甲虫种": "甲虫种",
    "两栖种": "两栖种",
    "甲壳种": "甲壳种",
    "铗角种": "铗角种",
    "章鱼种": "章鱼种",
    "蜘蛛种": "蜘蛛种",
    "？？？": "未知种类",
}

# RAG 提示词模板
RAG_PROMPT_TEMPLATE = """你是一个专业的《怪物猎人：荒野》怪物图鉴与狩猎顾问。请根据以下检索到的怪物/生物信息，回答玩家的问题。

【核心规则——务必遵守】：
玩家经常使用不精确的俗称、简称或外号来描述怪物，你必须根据检索结果自动匹配最接近的正式名称。例如：

| 玩家可能的说法 | 实际指的游戏怪物 | 推理逻辑 |
|--------------|----------------|--------|
| "雌火" / "雌火龙" | 【雌火龙】 | 简称/全称匹配 |
| "火龙" / "雄火" | 【火龙】 | 常用称呼匹配 |
| "丝瓜" / "恐暴" | 【恐暴龙】 | 外号/俗称匹配 |
| "咩咩子" / "灭尽" | 【灭尽龙】 | 玩家社区昵称匹配 |
| "太太" / "炎妃" | 【炎妃龙】 | 昵称/简称匹配 |
| "警察" / "金狮子" | 【金狮子】 | 外号匹配 |
| "角龙" / "黑角" | 【角龙】或【黑角龙】 | 简称匹配 |
| "烤鱼" / "熔岩龙" | 【熔岩龙】 | 外号匹配 |
| "青蛙" / "瓦斯蛙" | 【麻痹瓦斯蛙】 | 功能+外形匹配（环境生物） |
| "闪光虫" / "发光虫" | 【闪光翅虫】 | 功能描述匹配（环境生物） |

【数据类别说明】：
检索结果中可能包含两类数据：
- **Monster（大型怪物）**：拥有狩猎攻略的怪物，包含种类、生态、弱点、招式应对等信息。
- **EnvironmentCreature（环境生物）**：小型生态生物，包含外形特征和生态习性等信息。

【回答规范】：
1. 绝对不要说"没有找到你说的XX"或"图鉴中没有XX"——只要检索结果里有名称或特征相似的怪物/生物，就直接告诉玩家。
2. 回答时先用一句话确认匹配，例如"根据你的描述，对应的应该是【雌火龙】"。
3. 如果检索结果中有多个可能匹配的怪物（如雌火龙、火龙、樱火龙），把它们都列出来并说明区别。
4. 如果玩家问的是**打法/攻略**：
   - 先简要介绍怪物种类和特征（short_intro / ecology）
   - 再给出具体的狩猎攻略（hunting_guide），包括弱点、招式应对、部位破坏建议
   - 如有特殊个体/历战个体信息，一并列出
5. 如果玩家问的是**生态/背景**：
   - 优先用 ecology 字段的详细描述回答
   - 补充 short_intro 中的特征概述
6. 如果玩家问的是**环境生物**：
   - 简要描述其外形和功能（env_short_desc）
   - 如需深入了解，补充生态详情（env_long_desc）
7. 如果玩家问的怪物有历战个体（tempered_name）、历战王个体（tempered_king_name）、特殊个体（variant_name）或狂龙化形态（frenzy_name），在回答末尾提醒玩家存在这些变体。
8. 只有在检索结果中确实完全没有任何相关怪物/生物时，才回答"抱歉，我的狩猎笔记里暂时没有找到这个信息"。

【检索到的怪物/生物信息】：
{context}

【玩家的问题】：
{question}

【你的回答】："""


# ── 中文字符级分词器 ─────────────────────────────────────────
def chinese_char_tokenizer(text: str) -> list[str]:
    """中文字符级分词器，供 BM25 关键词检索使用。

    BM25 默认用空格分词，对中文无效。此分词器将每个汉字拆成独立 token，
    英文/数字保持连续，确保"雌火龙"这类关键词检索不会遗漏。
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
class MonsterDataRAG:
    """怪物图鉴 RAG 引擎。

    封装了怪物/环境生物数据的加载、向量数据库构建、混合检索（BM25 + 向量）、
    以及 LLM 生成回答的完整流程。可在节点函数中直接使用。

    属性:
        is_ready (bool): 是否已完成初始化。

    用法:
        rag = MonsterDataRAG()
        rag.setup()
        answer = rag.query("雌火龙怎么打？")
        # 或作为节点函数
        def monster_node(state):
            return {"answer": rag.query(state["question"])}
    """

    def __init__(
        self,
        data_path: str = "data/MonsterData.json",
        persist_dir: str = "./mh_monster_chroma_db",
        embedding_model: str = "BAAI/bge-small-zh-v1.5",
        bm25_k: int = 8,
        vector_k: int = 8,
    ):
        """初始化 MonsterDataRAG。

        参数:
            data_path: 怪物 JSON 数据文件路径。
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
        """获取已加载的怪物/生物 Document 列表。"""
        return self._docs

    # ── 初始化 ────────────────────────────────────────────

    def setup(self) -> "MonsterDataRAG":
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
        """加载怪物 JSON 数据，转为 LangChain Document 列表。"""
        print(f"正在加载怪物图鉴数据: {self._data_path} ...")
        documents = []
        with open(self._data_path, "r", encoding="utf-8") as f:
            data_list = json.load(f)

        for entry in data_list:
            name = entry.get("name", "")
            if not name or name == "---":
                continue

            category = entry.get("data_category", "")

            if category == "Monster":
                content = self._build_monster_content(entry)
            elif category == "EnvironmentCreature":
                content = self._build_env_creature_content(entry)
            else:
                continue

            if not content.strip():
                continue

            metadata = {
                "name": name,
                "data_category": category,
                "category_type": entry.get("category_type"),
                "short_intro": entry.get("short_intro"),
                "ecology": entry.get("ecology"),
                "hunting_guide": entry.get("hunting_guide"),
                "variant_name": entry.get("variant_name"),
                "variant_intro": entry.get("variant_intro"),
                "frenzy_name": entry.get("frenzy_name"),
                "tempered_name": entry.get("tempered_name"),
                "tempered_king_name": entry.get("tempered_king_name"),
                "env_short_desc": entry.get("env_short_desc"),
                "env_long_desc": entry.get("env_long_desc"),
            }
            # 去除 metadata 中的 None 值（Chroma 不接受 None）
            metadata = {k: v for k, v in metadata.items() if v is not None}

            documents.append(Document(page_content=content, metadata=metadata))

        self._docs = documents
        monster_count = sum(1 for d in documents if d.metadata.get("data_category") == "Monster")
        env_count = sum(1 for d in documents if d.metadata.get("data_category") == "EnvironmentCreature")
        print(f"成功加载 {len(documents)} 条怪物图鉴数据！(大型怪物: {monster_count}, 环境生物: {env_count})")

    @staticmethod
    def _build_monster_content(entry: dict) -> str:
        """构建大型怪物的 page_content，将关键字段拼接为语义丰富的文本块。"""
        parts = []

        name = entry.get("name", "")
        category_type = entry.get("category_type", "")
        short_intro = entry.get("short_intro", "")
        ecology = entry.get("ecology", "")
        hunting_guide = entry.get("hunting_guide", "")

        # 标题行
        parts.append(f"【{name}】是一种{category_type}大型怪物。")

        # 简短介绍
        if short_intro:
            parts.append(f"特征概述：{short_intro}")

        # 生态详情
        if ecology:
            parts.append(f"生态详情：{ecology}")

        # 狩猎攻略（核心）
        if hunting_guide:
            parts.append(f"狩猎攻略：{hunting_guide}")

        # 特殊变体
        variant_name = entry.get("variant_name")
        variant_intro = entry.get("variant_intro")
        if variant_name:
            extra = f"（{variant_intro}）" if variant_intro else ""
            parts.append(f"特殊个体：{variant_name}{extra}")

        frenzy_name = entry.get("frenzy_name")
        if frenzy_name:
            parts.append(f"狂龙化形态：{frenzy_name}")

        tempered_name = entry.get("tempered_name")
        if tempered_name:
            parts.append(f"历战个体：{tempered_name}")

        tempered_king_name = entry.get("tempered_king_name")
        if tempered_king_name:
            parts.append(f"历战王个体：{tempered_king_name}")

        return "。".join(parts) + "。"

    @staticmethod
    def _build_env_creature_content(entry: dict) -> str:
        """构建环境生物的 page_content。"""
        parts = []

        name = entry.get("name", "")
        short_intro = entry.get("short_intro", "")
        env_short_desc = entry.get("env_short_desc", "")
        env_long_desc = entry.get("env_long_desc", "")

        parts.append(f"【{name}】是一种环境生物。")

        # short_intro 可能在环境生物中也存在
        if short_intro:
            parts.append(f"特征概述：{short_intro}")

        if env_short_desc:
            parts.append(f"简述：{env_short_desc}")

        if env_long_desc:
            parts.append(f"生态详情：{env_long_desc}")

        return "。".join(parts) + "。"

    def _init_vectorstore(self):
        """构建或加载 Chroma 向量数据库。"""
        if _os.path.exists(self._persist_dir) and _os.listdir(self._persist_dir):
            print("检测到已有怪物向量数据库，直接加载...")
            self._vectorstore = Chroma(
                embedding_function=self._embeddings,
                persist_directory=self._persist_dir,
            )
        else:
            print("正在构建怪物向量数据库（首次运行需要一点时间）...")
            self._vectorstore = Chroma.from_documents(
                documents=self._docs,
                embedding=self._embeddings,
                persist_directory=self._persist_dir,
            )
        print("怪物向量数据库就绪！")

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
        """根据问题检索怪物/生物信息并生成回答。

        参数:
            question: 玩家的问题（如 "雌火龙怎么打？"）。

        返回:
            生成的回答字符串。

        异常:
            RuntimeError: 如果尚未调用 setup()。
        """
        if not self.is_ready:
            raise RuntimeError(
                "MonsterDataRAG 尚未初始化，请先调用 .setup() 方法。"
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
_default_instance: MonsterDataRAG | None = None


def get_monster_rag() -> MonsterDataRAG:
    """获取全局默认的 MonsterDataRAG 单例（惰性初始化）。"""
    global _default_instance
    if _default_instance is None:
        _default_instance = MonsterDataRAG()
        _default_instance.setup()
    return _default_instance


# ── 测试入口 ─────────────────────────────────────────────────
if __name__ == "__main__":
    rag = MonsterDataRAG()
    rag.setup()

    print("\n--- 怪猎怪物图鉴 Agent 已就绪 ---")
    while True:
        q = input("\n请输入你的问题 (输入 'quit' 退出): ")
        if q.lower() == "quit":
            break

        print("\n正在检索并生成回答...\n")
        response = rag.query(q)
        print(f"Agent: {response}")
