"""显式注册工具并固定名称映射，模型只能选择已注册的执行能力。"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable

from pydantic import BaseModel, TypeAdapter

from .contracts import ToolDescription, ToolExecutor, ToolName


@dataclass(frozen=True)
class ToolSpec:
    """声明工具的身份、输入输出模型、执行器及单次调用限制。"""

    name: str
    description: str
    arguments_model: type[BaseModel]
    result_model: type[BaseModel]
    executor: ToolExecutor
    version: str = "1"
    timeout_s: float = 30
    max_result_bytes: int = 8192

    def __post_init__(self):
        """注册前校验工具配置，要求输入输出模型均禁止未知字段。"""
        TypeAdapter(ToolName).validate_python(self.name)
        if not self.description or len(self.description) > 1000:
            raise ValueError("tool description must have 1..1000 characters")
        if not 0 < self.timeout_s <= 3600 or not 0 < self.max_result_bytes <= 65536:
            raise ValueError("invalid tool timeout or result size")
        for model in (self.arguments_model, self.result_model):
            if not issubclass(model, BaseModel) or model.model_config.get("extra") != "forbid":
                raise ValueError("tool schemas must be Pydantic models with extra='forbid'")

    def description_schema(self) -> ToolDescription:
        """生成模型可见的说明和参数结构，不暴露执行器对象。"""
        return ToolDescription(
            name=self.name, description=self.description,
            parameters=self.arguments_model.model_json_schema(),
        )


class ToolRegistry:
    """拒绝重名工具，构造后通过只读映射提供查询和说明接口。"""

    def __init__(self, tools: Iterable[ToolSpec] = ()):
        registered = {}
        for tool in tools:
            if tool.name in registered:
                raise ValueError(f"duplicate tool: {tool.name}")
            registered[tool.name] = tool
        # 固定名称到工具的映射，避免一次运行中途意外增删工具。
        self._tools = MappingProxyType(registered)

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def descriptions(self) -> list[ToolDescription]:
        return [tool.description_schema() for tool in self._tools.values()]
