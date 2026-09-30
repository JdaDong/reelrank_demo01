"""全局配置加载：config/settings.yaml + .env 环境变量。

用法::

    from reelrank.config import settings
    settings.tmdb.fetch.max_movies          # 嵌套属性访问
    settings.tmdb_api_key                   # 环境变量TMDB_API_KEY
    settings.path("data/warehouse")         # 相对项目根目录的绝对路径
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS_PATH = ROOT / "config" / "settings.yaml"


class AttrDict(dict):
    """支持属性访问的只读字典，用于嵌套配置访问。"""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:  # pragma: no cover - 配置缺字段时明确报错
            raise AttributeError(f"配置项不存在: {item}") from exc
        if isinstance(value, dict) and not isinstance(value, AttrDict):
            value = AttrDict(value)
            self[item] = value
        return value

    def __setattr__(self, key: str, value: Any) -> None:  # pragma: no cover
        self[key] = value


def _to_attrdict(node: Any) -> Any:
    if isinstance(node, dict):
        return AttrDict({k: _to_attrdict(v) for k, v in node.items()})
    if isinstance(node, list):
        return [_to_attrdict(v) for v in node]
    return node


class Settings(AttrDict):
    """项目配置对象，提供路径解析与环境变量快捷访问。"""

    @property
    def root(self) -> Path:
        return ROOT

    def path(self, *parts: str) -> Path:
        """返回相对项目根目录的绝对路径，父目录不存在时自动创建。"""
        target = ROOT.joinpath(*parts)
        if target.suffix == "":
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
        return target

    @property
    def tmdb_api_key(self) -> str:
        return os.getenv("TMDB_API_KEY", "").strip()

    @property
    def tmdb_access_token(self) -> str:
        return os.getenv("TMDB_ACCESS_TOKEN", "").strip()

    @property
    def tmdb_image_base(self) -> str:
        return os.getenv("TMDB_IMAGE_BASE", "").strip() or str(self["tmdb"]["image_base"])

    @property
    def has_tmdb_credential(self) -> bool:
        return bool(self.tmdb_api_key or self.tmdb_access_token)


def load_settings(path: str | Path | None = None) -> Settings:
    """加载 settings.yaml 并注入 .env 环境变量，返回全局配置对象。"""
    load_dotenv(ROOT / ".env", override=False)
    settings_path = Path(path) if path else DEFAULT_SETTINGS_PATH
    with open(settings_path, "r", encoding="utf-8") as fp:
        raw = yaml.safe_load(fp) or {}
    return Settings(_to_attrdict(raw))


settings = load_settings()
