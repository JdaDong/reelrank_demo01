"""TMDB API v3 客户端：限流、重试、本地缓存、断点续传。"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

import httpx

from reelrank.config import Settings, settings as global_settings
from reelrank.logging_utils import get_logger

logger = get_logger("reelrank.data.tmdb")


class TMDBError(RuntimeError):
    """TMDB 接口不可恢复的错误（凭证缺失、配额耗尽等）。"""


class TMDBClient:
    """线程安全的 TMDB 客户端。

    - 凭证：优先使用 v4 Bearer Token，其次 v3 api_key
    - 限流：全局最小请求间隔（默认 220ms，约 4.5 req/s）
    - 缓存：响应按 请求路径+参数 哈希落盘，重跑 ETL 不再打接口
    """

    def __init__(self, settings: Settings | None = None, cache_enabled: bool | None = None):
        self.settings = settings or global_settings
        self.base_url = str(self.settings.tmdb.api_base).rstrip("/")
        self.timeout = float(self.settings.tmdb.timeout_sec)
        self.max_retry = int(self.settings.tmdb.max_retry)
        self.interval = float(self.settings.tmdb.request_interval_ms) / 1000.0
        self.cache_enabled = bool(self.settings.tmdb.cache_enabled) if cache_enabled is None else cache_enabled
        self.cache_dir = self.settings.path("data/raw/cache")
        self._lock = threading.Lock()
        self._last_call_at = 0.0
        self._client = httpx.Client(timeout=self.timeout, follow_redirects=True)
        self._stats = {"api": 0, "cache": 0, "failed": 0}

    # ---------------- 凭证 ----------------
    @property
    def headers(self) -> dict[str, str]:
        token = self.settings.tmdb_access_token
        if token:
            return {"Authorization": f"Bearer {token}", "accept": "application/json"}
        return {"accept": "application/json"}

    def _auth_params(self) -> dict[str, Any]:
        token = self.settings.tmdb_access_token
        if token:
            return {}
        key = self.settings.tmdb_api_key
        if not key:
            raise TMDBError("缺少 TMDB 凭证：请在 .env 中配置 TMDB_API_KEY 或 TMDB_ACCESS_TOKEN")
        return {"api_key": key}

    # ---------------- 缓存 ----------------
    def _cache_path(self, path: str, params: dict[str, Any]) -> Path:
        payload = json.dumps({"p": path, "q": params}, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
        return self.cache_dir / digest[:2] / f"{digest}.json"

    def _read_cache(self, path: str, params: dict[str, Any]) -> Any | None:
        if not self.cache_enabled:
            return None
        target = self._cache_path(path, params)
        if not target.exists():
            return None
        try:
            return json.loads(target.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def _write_cache(self, path: str, params: dict[str, Any], data: Any) -> None:
        if not self.cache_enabled:
            return
        target = self._cache_path(path, params)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(target)

    # ---------------- 请求 ----------------
    def _throttle(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self.interval - (now - self._last_call_at)
            if wait > 0:
                time.sleep(wait)
            self._last_call_at = time.monotonic()

    def get(self, path: str, params: dict[str, Any] | None = None, use_cache: bool = True) -> Any | None:
        """GET 请求，返回解析后的 JSON；404 返回 None，其余错误重试后抛出。"""
        params = dict(params or {})
        params.update(self._auth_params())
        params.setdefault("language", self.settings.tmdb.language)
        params.setdefault("include_adult", str(self.settings.tmdb.include_adult).lower())

        if use_cache:
            cached = self._read_cache(path, {k: v for k, v in params.items() if k != "api_key"})
            if cached is not None:
                self._stats["cache"] += 1
                return cached

        url = f"{self.base_url}{path}"
        last_error: Exception | None = None
        for attempt in range(1, self.max_retry + 1):
            self._throttle()
            try:
                resp = self._client.get(url, params=params, headers=self.headers)
                if resp.status_code == 404:
                    self._stats["failed"] += 1
                    return None
                if resp.status_code == 429:
                    time.sleep(1.5 * attempt)
                    continue
                resp.raise_for_status()
                data = resp.json()
                self._stats["api"] += 1
                if use_cache:
                    self._write_cache(path, {k: v for k, v in params.items() if k != "api_key"}, data)
                return data
            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc
                logger.warning("TMDB 请求失败 (%s/%s) %s: %s", attempt, self.max_retry, path, exc)
                time.sleep(0.5 * attempt)

        self._stats["failed"] += 1
        logger.error("TMDB 请求最终失败: %s (%s)", path, last_error)
        return None

    def get_pages(self, path: str, params: dict[str, Any], max_pages: int) -> list[dict[str, Any]]:
        """分页拉取 list 接口，直到 max_pages 或总页数上限。"""
        rows: list[dict[str, Any]] = []
        for page in range(1, max_pages + 1):
            payload = self.get(path, {**params, "page": page})
            if not payload:
                break
            rows.extend(payload.get("results") or [])
            total_pages = int(payload.get("total_pages") or page)
            if page >= total_pages:
                break
        return rows

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    def close(self) -> None:
        self._client.close()
