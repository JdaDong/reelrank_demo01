"""DuckDB 数仓 ETL：原始 JSONL → Parquet → DuckDB 表（幂等重跑）。

入口::

    from reelrank.warehouse.etl import run_ingest
    run_ingest()   # 自动选择 TMDB 采集 / 样例快照兜底，再建仓
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from reelrank.config import Settings, settings as global_settings
from reelrank.logging_utils import get_logger

logger = get_logger("reelrank.warehouse.etl")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _explode(movies: list[dict[str, Any]], key: str, id_field: str, name_field: str, out_id: str, out_name: str) -> pd.DataFrame:
    records = []
    for movie in movies:
        for item in movie.get(key) or []:
            records.append(
                {
                    "movie_id": int(movie["id"]),
                    out_id: int(item.get(id_field) or 0),
                    out_name: item.get(name_field) or "",
                }
            )
    return pd.DataFrame(records, columns=["movie_id", out_id, out_name])


def build_frames(movies: list[dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, pd.DataFrame]:
    """把原始 JSONL 展开为数仓各表的 DataFrame。"""
    dim = pd.DataFrame(
        [
            {
                "movie_id": int(m["id"]),
                "title": m.get("title") or "",
                "original_title": m.get("original_title") or "",
                "overview": m.get("overview") or "",
                "tagline": m.get("tagline") or "",
                "release_date": m.get("release_date") or None,
                "year": m.get("year"),
                "runtime": int(m.get("runtime") or 0),
                "budget": int(m.get("budget") or 0),
                "revenue": int(m.get("revenue") or 0),
                "popularity": float(m.get("popularity") or 0.0),
                "vote_average": float(m.get("vote_average") or 0.0),
                "vote_count": int(m.get("vote_count") or 0),
                "original_language": m.get("original_language") or "",
                "adult": bool(m.get("adult")),
                "status": m.get("status") or "",
                "imdb_id": m.get("imdb_id") or "",
                "poster_path": m.get("poster_path") or "",
                "backdrop_path": m.get("backdrop_path") or "",
                "genre_names": [g["name"] for g in (m.get("genres") or [])],
                "keyword_names": [k["name"] for k in (m.get("keywords") or [])],
            }
            for m in movies
        ]
    )

    genre = _explode(movies, "genres", "id", "name", "genre_id", "genre_name")
    keyword = _explode(movies, "keywords", "id", "name", "keyword_id", "keyword_name")
    company = _explode(movies, "companies", "id", "name", "company_id", "company_name")

    cast_records = []
    for m in movies:
        for c in m.get("cast") or []:
            cast_records.append(
                {
                    "movie_id": int(m["id"]),
                    "person_id": int(c.get("id") or 0),
                    "person_name": c.get("name") or "",
                    "cast_order": int(c.get("order") or 999),
                    "character": c.get("character") or "",
                }
            )
    cast = pd.DataFrame(cast_records, columns=["movie_id", "person_id", "person_name", "cast_order", "character"])

    director_records = []
    for m in movies:
        for d in m.get("directors") or []:
            director_records.append(
                {
                    "movie_id": int(m["id"]),
                    "person_id": int(d.get("id") or 0),
                    "person_name": d.get("name") or "",
                }
            )
    director = pd.DataFrame(director_records, columns=["movie_id", "person_id", "person_name"])

    edge_records = []
    for e in edges:
        src = int(e["id"])
        for relation in ("similar", "recommended"):
            for pos, dst in enumerate(e.get(relation) or []):
                edge_records.append(
                    {"src_movie_id": src, "dst_movie_id": int(dst), "relation": relation, "rank_pos": pos}
                )
    sim_edge = pd.DataFrame(edge_records, columns=["src_movie_id", "dst_movie_id", "relation", "rank_pos"])
    if not sim_edge.empty:
        sim_edge = sim_edge.drop_duplicates(subset=["src_movie_id", "dst_movie_id", "relation"])

    return {
        "dim_movie": dim,
        "bridge_movie_genre": genre,
        "bridge_movie_keyword": keyword,
        "bridge_movie_cast": cast,
        "bridge_movie_director": director,
        "bridge_movie_company": company,
        "item_sim_edge": sim_edge,
    }


def _write_parquet(frames: dict[str, pd.DataFrame], parquet_dir: Path) -> dict[str, Path]:
    parquet_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, frame in frames.items():
        target = parquet_dir / f"{name}.parquet"
        if frame.empty:
            frame = frame.iloc[0:0]
        frame.to_parquet(target, index=False)
        paths[name] = target
    return paths


def run_etl(settings: Settings | None = None) -> dict[str, Any]:
    """读取 data/raw 下的 JSONL，构建 Parquet 中间层并重建 DuckDB 表。"""
    settings = settings or global_settings
    started = time.time()

    movies_path = settings.path("data/raw/movies.jsonl")
    edges_path = settings.path("data/raw/similar.jsonl")
    movies = _read_jsonl(movies_path)
    edges = _read_jsonl(edges_path)
    if not movies:
        raise RuntimeError(f"未找到影片原始数据：{movies_path}（请先执行采集或使用样例快照）")

    frames = build_frames(movies, edges)
    parquet_dir = settings.path(str(settings.warehouse.parquet_dir))
    paths = _write_parquet(frames, parquet_dir)

    schema_path = Path(__file__).with_name("schema.sql")
    db_path = settings.path(str(settings.warehouse.duckdb_path))
    con = duckdb.connect(str(db_path))
    try:
        con.execute(schema_path.read_text(encoding="utf-8"))
        counts: dict[str, int] = {}
        for name, target in paths.items():
            con.execute(
                f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM read_parquet('{target.as_posix()}')"
            )
            counts[name] = int(con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0])
        con.execute("CHECKPOINT")
    finally:
        con.close()

    raw_meta_path = settings.path("data/raw/meta.json")
    raw_meta = json.loads(raw_meta_path.read_text(encoding="utf-8")) if raw_meta_path.exists() else {}
    meta = {
        **raw_meta,
        "tables": counts,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(time.time() - started, 2),
    }
    meta_path = settings.path("data/warehouse/meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("数仓构建完成：%s 部影片 / %s 条关系边，耗时 %.2fs → %s", counts["dim_movie"], counts["item_sim_edge"], meta["elapsed_sec"], db_path)
    return meta


def ensure_raw(settings: Settings | None = None, refresh: bool = False) -> dict[str, Any]:
    """保证 data/raw 有数据：优先已有缓存 → TMDB 采集 → 样例快照兜底。"""
    settings = settings or global_settings
    movies_path = settings.path("data/raw/movies.jsonl")
    meta_path = settings.path("data/raw/meta.json")

    if movies_path.exists() and meta_path.exists() and not refresh:
        logger.info("复用已有原始数据：%s", movies_path)
        return json.loads(meta_path.read_text(encoding="utf-8"))

    if settings.has_tmdb_credential:
        from reelrank.data.fetch_tmdb import TMDBError, fetch_catalog

        try:
            return fetch_catalog(settings)
        except TMDBError as exc:
            logger.warning("TMDB 采集不可用，回退样例快照：%s", exc)
    else:
        logger.warning("未配置 TMDB 凭证（.env 中 TMDB_API_KEY），使用内置样例快照")

    from reelrank.data.seed_snapshot import materialize_seed_to_raw

    return materialize_seed_to_raw(settings)


def run_ingest(settings: Settings | None = None, refresh: bool = False) -> dict[str, Any]:
    """采集 + 建仓的一键入口。"""
    settings = settings or global_settings
    ensure_raw(settings, refresh=refresh)
    return run_etl(settings)


if __name__ == "__main__":
    print(json.dumps(run_ingest(), ensure_ascii=False, indent=2))
