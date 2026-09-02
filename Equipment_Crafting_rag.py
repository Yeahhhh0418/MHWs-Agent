"""
怪物猎人荒野 - 装备制作 RAG 模块
================================
提供 EquipmentCraftingRAG 类，封装装备制作数据的检索增强生成（RAG）。
支持在节点函数中直接实例化和调用。

用法示例:
    from Equipment_Crafting_rag import EquipmentCraftingRAG

    equip_rag = EquipmentCraftingRAG()
    equip_rag.setup()
    answer = equip_rag.query("炎尾龙的头怎么制作？")
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

from agent.chatllm import llm

# ── 常量 ─────────────────────────────────────────────────────
RAG_PROMPT_TEMPLATE = """你是一个专业的《怪物猎人：荒野》攻略助手。请根据以下检索到的攻略信息，回答玩家的问题。

【核心规则——务必遵守】：
玩家经常使用不精确的俗称来描述装备，你必须根据检索结果自动匹配最接近的正式装备名。例如：

| 玩家可能的说法 | 实际指的游戏装备 | 推理逻辑 |
|--------------|----------------|--------|
| "炎尾龙的头" | 【炎尾龙头盔】 | "炎尾龙"是怪物名，"头"对应头部防具→头盔 |
| "缠蛙的衣服" | 【缠蛙铠甲】 | "缠蛙"是怪物名，"衣服"对应胸部防具→铠甲 |
| "辟兽的手" | 【辟兽腕甲】 | "辟兽"是怪物名，"手"对应腕部防具→腕甲 |
| "沙海龙的腰" | 【沙海龙腰甲】 | "沙海龙"是怪物名，"腰"对应腰部防具→腰甲 |
| "沼喷龙的腿" | 【沼喷龙护腿】 | "沼喷龙"是怪物名，"腿"对应腿部防具→护腿 |
| "护火龙的胸" | 【护火龙铠甲】 | "护火龙"是怪物名，"胸"对应胸部防具→铠甲 |
| "希望面具" | 【希望面具】 | 玩家直接说出完整装备名，直接匹配 |

【回答规范】：
1. 绝对不要说"没有找到你说的XX"或"攻略信息中没有XX"——只要检索结果里有名称相似的装备，就直接告诉玩家它叫什么、怎么制作。
2. 回答时先用一句话确认匹配，例如"根据你的描述，对应的装备是【炎尾龙头盔】"，然后给出制作信息。
3. 如果检索结果中有多个可能匹配的装备（如炎尾龙头盔、炎尾龙头盔α、炎尾龙头盔β），把它们都列出来并说明区别。
4. 如果检索结果中同时包含同名武器和防具，优先列出防具再列武器（或反过来），让玩家选择。
5. 只有在检索结果中确实完全没有任何相关装备时，才回答"抱歉，我的狩猎笔记里暂时没有找到这个信息"。

【检索到的攻略信息】：
{context}

【玩家的问题】：
{question}

【你的回答】："""


# ── 主类 ─────────────────────────────────────────────────────
class EquipmentCraftingRAG:
    """装备制作 RAG 引擎。

    封装了装备制作数据的加载、向量数据库构建、语义检索、
    以及 LLM 生成回答的完整流程。可在节点函数中直接使用。

    属性:
        is_ready (bool): 是否已完成初始化。

    用法:
        rag = EquipmentCraftingRAG()
        rag.setup()
        answer = rag.query("炎尾龙的头盔怎么制作？")
        # 或作为节点函数
        def equip_node(state):
            return {"answer": rag.query(state["question"])}
    """

    def __init__(
        self,
        data_path: str = "data/Equipment_Crafting_Table.json",
        persist_dir: str = "./mh_equipmentcrafting_chroma_db",
        embedding_model: str = "BAAI/bge-small-zh-v1.5",
        top_k: int = 3,
    ):
        """初始化 EquipmentCraftingRAG。

        参数:
            data_path: 装备制作 JSON 数据文件路径。
            persist_dir: Chroma 向量数据库持久化目录。
            embedding_model: HuggingFace embedding 模型名称。
            top_k: 语义检索返回的最相关文档数。
        """
        self._data_path = data_path
        self._persist_dir = persist_dir
        self._embedding_model = embedding_model
        self._top_k = top_k

        self.is_ready = False
        self._chain = None
        self._retriever = None
        self._vectorstore = None
        self._docs = None
        self._llm = None

    # ── 属性 ──────────────────────────────────────────────

    @property
    def retriever(self):
        """获取向量检索器。"""
        return self._retriever

    @property
    def documents(self):
        """获取已加载的装备 Document 列表。"""
        return self._docs

    # ── 初始化 ────────────────────────────────────────────

    def setup(self) -> "EquipmentCraftingRAG":
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
        """加载装备制作 JSON 数据，转为 LangChain Document 列表。"""
        print(f"正在加载怪猎装备知识库: {self._data_path} ...")
        documents = []
        with open(self._data_path, "r", encoding="utf-8") as f:
            data_list = json.load(f)

        for data in data_list:
            meta = data.get("metadata", {})
            # ChromaDB 不接受空列表，将其转为 None
            sanitized = {}
            for k, v in meta.items():
                if isinstance(v, list) and len(v) == 0:
                    sanitized[k] = None
                else:
                    sanitized[k] = v

            doc = Document(
                page_content=data["content"],
                metadata=sanitized,
            )
            documents.append(doc)

        self._docs = documents
        print(f"成功加载 {len(documents)} 条装备数据！")

    def _init_vectorstore(self):
        """构建或加载 Chroma 向量数据库。"""
        if _os.path.exists(self._persist_dir) and _os.listdir(self._persist_dir):
            print("检测到已有装备向量数据库，直接加载...")
            self._vectorstore = Chroma(
                embedding_function=self._embeddings,
                persist_directory=self._persist_dir,
            )
        else:
            print("正在构建装备向量数据库（首次运行需要一点时间）...")
            self._vectorstore = Chroma.from_documents(
                documents=self._docs,
                embedding=self._embeddings,
                persist_directory=self._persist_dir,
            )
        print("装备向量数据库就绪！")

    def _init_retriever(self):
        """创建语义向量检索器。"""
        self._retriever = self._vectorstore.as_retriever(
            search_kwargs={"k": self._top_k}
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
        """根据问题检索装备制作信息并生成回答。

        参数:
            question: 玩家的问题（如 "炎尾龙的头怎么制作？"）。

        返回:
            生成的回答字符串。

        异常:
            RuntimeError: 如果尚未调用 setup()。
        """
        if not self.is_ready:
            raise RuntimeError(
                "EquipmentCraftingRAG 尚未初始化，请先调用 .setup() 方法。"
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
_default_instance: EquipmentCraftingRAG | None = None


def get_equipment_rag() -> EquipmentCraftingRAG:
    """获取全局默认的 EquipmentCraftingRAG 单例（惰性初始化）。"""
    global _default_instance
    if _default_instance is None:
        _default_instance = EquipmentCraftingRAG()
        _default_instance.setup()
    return _default_instance


# ── 测试入口 ─────────────────────────────────────────────────
if __name__ == "__main__":
    rag = EquipmentCraftingRAG()
    rag.setup()

    print("\n--- 怪猎攻略 Agent 已就绪 ---")
    while True:
        q = input("\n请输入你的问题 (输入 'quit' 退出): ")
        if q.lower() == "quit":
            break

        print("\n正在检索并生成回答...\n")
        response = rag.query(q)
        print(f"Agent: {response}")