"""定义跨恢复保留的执行策略；业务调用上限与图递归上限分别控制。"""

from pydantic import Field

from .contracts import Schema


class RunPolicy(Schema):
    """配置模型、工具、澄清和上下文预算，构造时校验取值范围。"""

    max_model_calls: int = Field(default=8, strict=True, ge=1, le=1000)
    max_tool_calls: int = Field(default=8, strict=True, ge=0, le=1000)
    max_action_repairs: int = Field(default=2, strict=True, ge=0, le=20)
    max_plan_repairs: int = Field(default=2, strict=True, ge=0, le=20)
    max_questions: int = Field(default=8, strict=True, ge=1, le=100)
    max_answer_repairs: int = Field(default=2, strict=True, ge=0, le=20)
    # 问题等待按现实时间计时，和下方的累计执行时间预算互不混用。
    question_timeout_s: float = Field(default=1800, gt=0, le=604800)
    max_answer_bytes: int = Field(default=8192, strict=True, ge=1, le=65536)
    deadline_seconds: float = Field(default=120, gt=0, le=3600)
    call_timeout_s: float = Field(default=30, gt=0, le=3600)
    # 此处只限制上下文字符数；真实模型适配器仍需校验分词后的令牌数量。
    max_context_chars: int = Field(default=16000, strict=True, ge=1000, le=1000000)
    max_history_messages: int = Field(default=12, strict=True, ge=0, le=100)
    max_output_tokens: int = Field(default=512, strict=True, ge=1, le=32768)

    @property
    def recursion_limit(self) -> int:
        """按节点转换次数估算框架递归保险，不替代模型、工具及问答的业务计数。"""
        return 5 * self.max_model_calls + 3 * self.max_questions + 10
