"""
HTTP 入口：把 Agent 包成接口给外部调。
只做三件事 —— 收参数、调 Agent、把错误翻译成人话；业务逻辑一概不写在这里。
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage

from app.agent.graph import graph
from app.agent.nodes.retrieve import _get_rag  # ⚠️ 私有函数，见 §4.2
from app.core.logger import get_logger
from app.core.metrics import _ledger, format_cost_line, summary
from app.server.schemas import AskRequest, AskResponse

logger = get_logger("server")

# 四个库的名字：启动预热时全建一遍
_DBS = ("equipment", "weapon", "material", "monster")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    [WHY] RAG 是懒加载的 —— 第一次问到某个库才 setup()，而 setup() 要加载 BGE 向量模型、
          建检索器，很贵。不预热的话，第一个用户得干等我们的启动成本。
    [FIX] 启动时把四个库都建好；顺带消灭"两个请求同时首次问同一个库、各自 setup() 一遍"
          这个竞态（_rag_cache 没有锁，见 §4.4）。
    """
    # 启动阶段：四个库全建一遍，别让第一个用户等
    for db in _DBS:
        _get_rag(db)
        logger.info("预热完成", extra={"db": db})
    # 关停阶段：现在没有要收尾的东西，留着占位
    yield


app = FastAPI(title="MHWI 攻略问答", lifespan=lifespan)


@app.get("/health")
def health():
    """探活：什么也不干，回一句"我还活着"；负载均衡和监控靠它判断进程还能不能用。"""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """
    [WHY] 必须是 def，不能写成 async def —— graph.invoke() 是同步阻塞的，一次要跑
          11 秒起步（最慢 5 分钟）。写成 async def 会让它占住事件循环那唯一一条道，
          从第 2 个请求开始全部排队，现象是"服务像是挂了"。
    [FIX] 用 def，FastAPI 会自动把它丢进线程池，一个请求占一条线程。
    """
    # 掐表：从进图到出答案，最后跟 token 一起报给用户
    started = time.perf_counter()

    try:
        # 进图；同步阻塞就发生在这一行
        state = graph.invoke({"messages": [HumanMessage(content=req.question)]})
    except Exception as e:  # 顶层兜底：完整异常进日志，只把摘要给用户
        # 堆栈和路径只进日志：漏给用户等于把密钥和用户名送出去
        logger.error("问答失败", extra={"error": f"{type(e).__name__}: {e}"})
        raise HTTPException(status_code=500, detail="处理失败，请稍后再试") from e

    # 这次问答的总耗时
    elapsed = time.perf_counter() - started
    # 编号是 route 节点发的；出问题时按它从日志里捞整条链路
    rid = state.get("request_id", "")

    # 先把账抄走再撕掉 —— 账本只增不减，服务跑久了内存一直涨（§4.4）
    usage = summary(rid)
    _ledger.pop(rid, None)

    # 组装响应；final_answer 理论上一定有，兜一句免得返回 null
    return AskResponse(
        answer=state.get("final_answer", "（没有拿到答案）"),
        request_id=rid,
        degraded=state.get("degraded", False),
        degrade_reason=state.get("degrade_reason", ""),
        cost=format_cost_line(usage, elapsed),
    )
