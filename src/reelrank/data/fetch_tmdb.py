"""TMDB 目录抓取：list 接口收集 id → 详情接口补全 → 落盘 JSONL。

产物：
- data/raw/movies.jsonl   每部影片一行（含 genres/keywords/cast/directors/companies）
- data/raw/similar.jsonl  每部影片一行（similar + recommendations 关系边）
- data/raw/meta.json      采集元信息（来源、数量、时间）
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from reelrank.config import Settings, settings as global_settings
from reelrank.data.tmdb_client import TMDBClient, TMDBError
from reelrank.logging_utils import get_logger

logger = get_logger("reelrank.data.fetch")


def _year(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(str(value)[:4])
    except ValueError:
        return None


def normalize_movie(detail: dict[str, Any], similar_limit: int = 20) -> tuple[dict[str, Any], dict[str, Any]]:
    """把 TMDB 详情响应标准化为 (影片记录, 关系边记录)。"""
    credits = detail.get("credits") or {}
    keywords_blob = detail.get("keywords") or {}
    keyword_items = keywords_blob.get("keywords") or keywords_blob.get("results") or []
    cast_items = (credits.get("cast") or [])[:12]
    crew_items = credits.get("crew") or []

    movie = {
        "id": int(detail["id"]),
        "title": detail.get("title") or detail.get("original_title") or "",
        "original_title": detail.get("original_title") or "",
        "overview": detail.get("overview") or "",
        "tagline": detail.get("tagline") or "",
        "release_date": detail.get("release_date") or None,
        "year": _year(detail.get("release_date")),
        "runtime": detail.get("runtime") or 0,
        "budget": int(detail.get("budget") or 0),
        "revenue": int(detail.get("revenue") or 0),
        "popularity": float(detail.get("popularity") or 0.0),
        "vote_average": float(detail.get("vote_average") or 0.0),
        "vote_count": int(detail.get("vote_count") or 0),
        "original_language": detail.get("original_language") or "",
        "adult": bool(detail.get("adult")),
        "status": detail.get("status") or "",
        "imdb_id": detail.get("imdb_id") or "",
        "poster_path": detail.get("poster_path") or "",
        "backdrop_path": detail.get("backdrop_path") or "",
        "genres": [{"id": int(g["id"]), "name": g["name"]} for g in (detail.get("genres") or [])],
        "keywords": [{"id": int(k["id"]), "name": k["name"]} for k in keyword_items],
        "cast": [
            {
                "id": int(c.get("id") or 0),
                "name": c.get("name") or "",
                "order": int(c.get("order") or 999),
                "character": c.get("character") or "",
            }
            for c in cast_items
            if c.get("name")
        ],
        "directors": [
            {"id": int(p.get("id") or 0), "name": p.get("name") or ""}
            for p in crew_items
            if p.get("job") == "Director" and p.get("name")
        ],
        "companies": [
            {"id": int(c.get("id") or 0), "name": c.get("name") or ""}
            for c in (detail.get("production_companies") or [])
        ],
    }

    edges = {
        "id": movie["id"],
        "similar": [int(s["id"]) for s in ((detail.get("similar") or {}).get("results") or [])][:similar_limit],
        "recommended": [
            int(s["id"]) for s in ((detail.get("recommendations") or {}).get("results") or [])
        ][:similar_limit],
    }
    return movie, edges


def _collect_movie_ids(client: TMDBClient, settings: Settings) -> list[int]:
    """从 popular / top_rated / discover / search 多路收集影片 id。"""
    cfg = settings.tmdb.fetch
    ids: list[int] = []
    seen: set[int] = set()

    def add(rows: Iterable[dict[str, Any]]) -> None:
        for row in rows:
            mid = row.get("id")
            if mid and mid not in seen:
                seen.add(int(mid))
                ids.append(int(mid))

    add(client.get_pages("/movie/popular", {}, int(cfg.popular_pages)))
    logger.info("popular 采集完成，累计 %s 部", len(ids))
    add(client.get_pages("/movie/top_rated", {}, int(cfg.top_rated_pages)))
    logger.info("top_rated 采集完成，累计 %s 部", len(ids))

    for year in range(int(cfg.discover_year_start), int(cfg.discover_year_end) + 1):
        rows = client.get_pages(
            "/discover/movie",
            {"primary_release_year": year, "sort_by": "vote_count.desc"},
            int(cfg.discover_pages_per_year),
        )
        add(rows)
        if len(ids) >= int(cfg.max_movies):
            break
    logger.info("discover 采集完成，累计 %s 部", len(ids))

    for query in list(cfg.search_queries):
        rows = client.get_pages("/search/movie", {"query": query}, int(cfg.search_pages_per_query))
        add(rows)
        if len(ids) >= int(cfg.max_movies):
            break
    logger.info("search 采集完成，累计 %s 部", len(ids))
    return ids[: int(cfg.max_movies)]


def _fetch_details(client: TMDBClient, movie_ids: list[int], settings: Settings) -> tuple[list[dict], list[dict]]:
    cfg = settings.tmdb.fetch
    movies: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    similar_limit = int(cfg.similar_graph_limit)

    def one(mid: int) -> tuple[dict, dict] | None:
        detail = client.get(
            f"/movie/{mid}",
            {"append_to_response": "credits,keywords,similar,recommendations"},
        )
        if not detail or not detail.get("title"):
            return None
        return normalize_movie(detail, similar_limit)

    workers = max(1, int(cfg.detail_concurrency))
    logger.info("开始补全 %s 部影片详情（并发 %s）", len(movie_ids), workers)
    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for idx, result in enumerate(pool.map(one, movie_ids), start=1):
            if result:
                movies.append(result[0])
                edges.append(result[1])
            if idx % 200 == 0:
                logger.info("详情进度 %s/%s，耗时 %.1fs", idx, len(movie_ids), time.time() - started)
    logger.info("详情补全完成：%s 部，耗时 %.1fs", len(movies), time.time() - started)
    return movies, edges


def _dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def fetch_catalog(settings: Settings | None = None, client: TMDBClient | None = None) -> dict[str, Any]:
    """抓取 TMDB 目录并落盘；无凭证时抛出 TMDBError 由上层回退样例快照。"""
    settings = settings or global_settings
    owns_client = client is None
    client = client or TMDBClient(settings)

    try:
        if not settings.has_tmdb_credential:
            raise TMDBError("缺少 TMDB 凭证")
        movie_ids = _collect_movie_ids(client, settings)
        movies, edges = _fetch_details(client, movie_ids, settings)
    finally:
        if owns_client:
            client.close()

    if not movies:
        raise TMDBError("TMDB 未返回任何影片数据")

    movies_path = settings.path("data/raw/movies.jsonl")
    edges_path = settings.path("data/raw/similar.jsonl")
    _dump_jsonl(movies_path, movies)
    _dump_jsonl(edges_path, edges)

    meta = {
        "source": "tmdb_api",
        "movie_count": len(movies),
        "edge_count": len(edges),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "client_stats": client.stats,
    }
    meta_path = settings.path("data/raw/meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("TMDB 采集落盘：%s 部影片 → %s", len(movies), movies_path)
    return meta


def main() -> int:
    try:
        meta = fetch_catalog()
    except TMDBError as exc:
        logger.error("采集失败：%s（将使用内置样例快照）", exc)
        return 1
    logger.info("采集完成：%s", meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
