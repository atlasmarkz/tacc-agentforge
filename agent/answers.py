"""校验结构化澄清回答，只允许修改本次问题指定的约束字段。"""

import json

from pydantic import TypeAdapter, ValidationError

from .contracts import Amount, Issue, PendingQuestion, ResumeRequest
from .policies import RunPolicy


def validate_answer(question: PendingQuestion, reply: ResumeRequest,
                    constraints: dict, policy: RunPolicy) -> Issue | None:
    """通过时返回 None，否则返回边界校验问题；完整业务规则仍由方案校验器负责。"""
    if len(json.dumps(reply.answer, ensure_ascii=False).encode("utf-8")) > policy.max_answer_bytes:
        return Issue(code="answer_too_large", message="Answer exceeds the configured size limit.")
    # 要求字段集合完全一致，既防止遗漏回答，也防止额外字段越过问题的修改范围。
    if set(reply.answer) != set(question.required_fields):
        return Issue(code="invalid_answer_fields", message="Answer exactly the requested constraint fields.")
    for name, value in reply.answer.items():
        if value is None or (isinstance(value, str) and not value.strip()) or value == [] or value == {}:
            return Issue(code="empty_answer", message=f"Provide a nonempty value for {name}.")
        if name == "purchase_cap":
            # 金额保持为非负十进制字符串，避免浮点误差和隐式类型转换。
            try:
                TypeAdapter(Amount).validate_python(value, strict=True)
            except ValidationError:
                return Issue(code="invalid_budget", message="purchase_cap must be a nonnegative decimal string.")
        elif name == "currency":
            if not isinstance(value, str) or len(value) != 3 or not value.isascii() or not value.isupper() or not value.isalpha():
                return Issue(code="invalid_currency", message="currency must be a three-letter uppercase code.")
        elif name in constraints and constraints[name] is not None and type(value) is not type(constraints[name]):
            return Issue(code="invalid_answer_type", message=f"Preserve the existing type for {name}.")
    return None
