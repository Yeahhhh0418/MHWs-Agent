"""HTTP 接口的请求 / 响应形状 —— 给外部看的合同，改动要当心。"""

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """
    [WHY] question 必须限长：这个接口每问一次都在烧模型的钱，
          不限长就是给人留一个白嫖 token 的口子。
    """

    question: str = Field(min_length=1, max_length=200, description="用户的问题")


class AskResponse(BaseModel):
    """
    [WHY] 只放"用户要的 + 出问题时要报的"。内部的 collected_info 和 decision_trace
          一概不进响应 —— 它们体量大，还会把检索到的资料原文带给外部。
    """

    # 给用户看的答案
    answer: str
    # 出问题时按它捞整条日志
    request_id: str
    # 这次问答有没有哪一步降级过（检索失败、模型超时都会置上）
    degraded: bool
    # 降级原因；没降级就是空串
    degrade_reason: str
    # 给调试看的一行：耗时 / token / 调用次数
    cost: str
