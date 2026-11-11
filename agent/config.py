"""Agent 专用 YAML 配置读取；不创建模型、数据库或网络连接。"""

from copy import deepcopy
from dataclasses import dataclass
import os
from pathlib import Path
import re

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/agent.yaml"


def merge(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


@dataclass(frozen=True)
class AgentConfig:
    path: Path
    profile: str
    values: dict

    @property
    def root(self):
        return self.path.parent.parent if self.path.parent.name == "configs" else self.path.parent

    def resolve_path(self, value):
        if not isinstance(value, (str, Path)) or not str(value):
            raise ValueError("Configured path must be nonempty")
        expanded = os.path.expandvars(str(value))
        if re.search(r"\$[A-Za-z_{]", expanded):
            raise ValueError("Configured path contains an undefined environment variable")
        path = Path(expanded).expanduser()
        return (path if path.is_absolute() else self.root / path).resolve()


def load_config(path=None, profile=None, *, default_profile=None):
    import yaml

    class UniqueLoader(yaml.SafeLoader):
        pass

    def mapping(loader, node):
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node)
            if key in result:
                raise ValueError(f"Duplicate configuration key: {key}")
            result[key] = loader.construct_object(value_node)
        return result
    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    path = Path(path or os.environ.get("AGENT_CONFIG") or DEFAULT_CONFIG).expanduser().resolve()
    try:
        value = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML configuration: {path}") from exc
    if (not isinstance(value, dict) or type(value.get("version")) is not int or value["version"] != 1
            or set(value) - {"version", "default_profile", "defaults", "profiles"}):
        raise ValueError("Unsupported agent configuration")
    profiles = value.get("profiles", {})
    defaults = value.get("defaults", {})
    if not isinstance(profiles, dict) or not isinstance(defaults, dict):
        raise ValueError("Configuration defaults and profiles must be mappings")
    selected = profile or os.environ.get("AGENT_PROFILE") or default_profile or value.get("default_profile")

    def resolve(name, visited):
        if name in visited:
            raise ValueError("Configuration profile inheritance contains a cycle")
        if name not in profiles or not isinstance(profiles[name], dict):
            raise ValueError(f"Unknown or invalid configuration profile: {name}")
        changes = dict(profiles[name])
        parent = changes.pop("extends", None)
        return merge(resolve(parent, visited | {name}) if parent else defaults, changes)
    values = resolve(selected, set())
    return AgentConfig(path, selected, values)
