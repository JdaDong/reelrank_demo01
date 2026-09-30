"""FastAPI 服务：搜索 / 推荐 / 漏斗调试 / 广告看板 / 健康检查。"""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from reelrank.config import Settings, settings as global_settings
from reelrank.logging_utils import configure, get_logger
from reelrank.serving.funnel import run_funnel
from reelrank.serving.schemas import RankResponse
from reelrank.serving.state import RankState, get_state

logger = get_logger("reelrank.serving.api")

STATE: RankState | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global STATE
    configure(str(app.state.settings.project.log_level) if hasattr(app.state, "settings") else "INFO")
    STATE = get_state()
    logger.info("ReelRank 服务就绪：%s 部影片 / %s 个用户", len(STATE.item_raw), len(STATE.profiles))
    yield


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or global_settings
    app = FastAPI(title="ReelRank", version="0.1.0", description="TMDB 搜索/推荐/广告四层漏斗排序系统", lifespan=lifespan)
    app.state.settings = settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _state() -> RankState:
        if STATE is None:
            raise HTTPException(status_code=503, detail="服务尚未就绪，索引与模型还在加载")
        return STATE

    def _cached(key: str):
        return _state().cache_get(key)

    # ---------------- 健康检查与元信息 ----------------
    @app.get("/health", summary="健康检查")
    def health() -> dict[str, Any]:
        state = _state()
        return {
            "status": "ok",
            "movies": len(state.item_raw),
            "users": len(state.profiles),
            "ads": len(state.ad_index.ads),
            "coarse_auc": state.coarse.train_auc,
            "fine_auc": state.fine.valid_auc,
            "vector_explained_variance": round(state.vector.explained_variance, 4),
        }

    @app.get("/api/stats", summary="系统配置与产物信息")
    def stats() -> dict[str, Any]:
        state = _state()
        settings = app.state.settings
        meta_path = state.settings.path("artifacts/train_meta.json")
        train_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        warehouse_meta_path = state.settings.path("data/warehouse/meta.json")
        warehouse_meta = json.loads(warehouse_meta_path.read_text(encoding="utf-8")) if warehouse_meta_path.exists() else {}
        return {
            "funnel": {
                "recall": dict(settings.funnel.recall),
                "coarse": dict(settings.funnel.coarse),
                "fine": {"topk": settings.funnel.fine.topk, "weights": {
                    "pctr": settings.funnel.fine.w_pctr,
                    "quality": settings.funnel.fine.w_quality,
                    "freshness": settings.funnel.fine.w_freshness,
                }},
                "rerank": dict(settings.funnel.rerank),
            },
            "ads": {"enabled": settings.ads.enabled, "slots": list(settings.ads.slots), "gsp": settings.ads.gsp_enabled},
            "data_source": warehouse_meta.get("source"),
            "behavior": train_meta.get("positive_rate"),
            "train": train_meta.get("coarse") and {"coarse_auc": train_meta["coarse"].get("train_auc"),
                                                   "fine_valid_auc": train_meta.get("fine", {}).get("valid_auc")},
            "top_features": train_meta.get("fine", {}).get("top_features", [])[:6],
        }

    @app.get("/api/users", summary="可选用户列表（前端切换用户）")
    def list_users(limit: int = Query(50, ge=1, le=500)) -> dict[str, Any]:
        state = _state()
        users = sorted(state.profiles.keys())[:limit]
        return {
            "users": [
                {
                    "user_id": user_id,
                    "top_genres": state.profiles[user_id].get("top_genres", []),
                    "click_rate": round(float(state.profiles[user_id].get("user_click_rate", 0.0)), 3),
                    "history_size": len(state.user_history.get(user_id, [])),
                }
                for user_id in users
            ]
        }

    @app.get("/api/movies/{movie_id}", summary="影片详情卡")
    def movie_detail(movie_id: int) -> dict[str, Any]:
        state = _state()
        if movie_id not in state.item_raw:
            raise HTTPException(status_code=404, detail="影片不存在")
        card = state.movie_card(movie_id)
        card["features"] = {k: round(float(v), 4) for k, v in state.item_features[movie_id].items()}
        card["similar"] = [
            {"movie_id": mid, "title": state.title_of(mid), "score": round(float(score), 4)}
            for mid, score in state.itemcf.neighbors_of(movie_id, topk=6)
        ]
        return card

    # ---------------- 排序主链路 ----------------
    @app.get("/api/search", response_model=RankResponse, summary="搜索（走完整四层漏斗）")
    def search(
        query: str = Query(..., min_length=1, description="搜索词"),
        user_id: int | None = Query(None),
        topk: int | None = Query(None, ge=1, le=100),
        session_id: str | None = Query(None),
        ads: bool | None = Query(None, description="是否启用广告"),
        trace: bool | None = Query(None, description="是否返回 FunnelTrace"),
    ) -> RankResponse:
        state = _state()
        key = f"search:{query}:{user_id}:{topk}:{ads}:{trace}"
        cached = _cached(key)
        if cached is not None:
            return cached
        response = run_funnel(
            state, scene="search", query=query, user_id=user_id, session_id=session_id,
            topk=topk, ads_enabled=ads, with_trace=trace, settings=app.state.settings,
        )
        state.cache_put(key, response)
        return response

    @app.get("/api/recommend", response_model=RankResponse, summary="个性化推荐（同一漏斗）")
    def recommend(
        user_id: int | None = Query(None),
        topk: int | None = Query(None, ge=1, le=100),
        session_id: str | None = Query(None),
        ads: bool | None = Query(None),
        trace: bool | None = Query(None),
    ) -> RankResponse:
        state = _state()
        key = f"recommend:{user_id}:{topk}:{ads}:{trace}"
        cached = _cached(key)
        if cached is not None:
            return cached
        response = run_funnel(
            state, scene="recommend", user_id=user_id, session_id=session_id,
            topk=topk, ads_enabled=ads, with_trace=trace, settings=app.state.settings,
        )
        state.cache_put(key, response)
        return response

    @app.post("/api/funnel/debug", response_model=RankResponse, summary="漏斗调试：可覆盖各层截断数")
    def funnel_debug(payload: dict[str, Any]) -> RankResponse:
        state = _state()
        scene = str(payload.get("scene") or ("search" if payload.get("query") else "recommend"))
        if scene not in ("search", "recommend"):
            raise HTTPException(status_code=400, detail="scene 只能是 search 或 recommend")
        return run_funnel(
            state,
            scene=scene,
            query=payload.get("query"),
            user_id=payload.get("user_id"),
            session_id=payload.get("session_id"),
            topk=payload.get("topk"),
            ads_enabled=payload.get("ads_enabled"),
            recall_limit=payload.get("recall_limit"),
            coarse_topk=payload.get("coarse_topk"),
            fine_topk=payload.get("fine_topk"),
            with_trace=True,
            settings=app.state.settings,
        )

    # ---------------- 广告 ----------------
    @app.get("/api/ads/board", summary="广告看板：广告主出价、预算与消耗")
    def ads_board() -> dict[str, Any]:
        state = _state()
        board = state.ad_index.advertiser_board()
        return {
            "date": state.ad_index.budget_date,
            "slots": list(app.state.settings.ads.slots),
            "advertisers": board,
            "total_spent": round(sum(item["spent"] for item in board), 2),
            "total_budget": round(sum(item["daily_budget"] for item in board), 2),
        }

    @app.post("/api/ads/reset", summary="重置当日广告预算（演示用）")
    def ads_reset() -> dict[str, Any]:
        state = _state()
        state.ad_index.spent = {str(ad.ad_id): 0.0 for ad in state.ad_index.ads}
        state.ad_index.persist_budget()
        state.sessions.clear()
        logger.info("广告预算与会话频次已重置")
        return {"status": "ok", "ads": len(state.ad_index.ads)}

    @app.get("/api/ads", summary="广告库明细")
    def ads_list(limit: int = Query(50, ge=1, le=500)) -> dict[str, Any]:
        state = _state()
        ads = state.ad_index.ads[:limit]
        return {
            "ads": [
                {
                    "ad_id": ad.ad_id,
                    "advertiser_id": ad.advertiser_id,
                    "advertiser_name": ad.advertiser_name,
                    "movie_id": ad.movie_id,
                    "title": ad.title,
                    "bid": ad.bid,
                    "daily_budget": ad.daily_budget,
                    "spent": round(float(state.ad_index.spent.get(str(ad.ad_id), 0.0)), 2),
                    "freq_cap": ad.freq_cap,
                    "targeting": ad.targeting,
                }
                for ad in ads
            ]
        }

    # ---------------- 前端静态资源（构建后存在时挂载） ----------------
    dist = Path(settings.root) / "web" / "dist"
    if dist.exists():
        from fastapi.responses import FileResponse
        from fastapi.staticfiles import StaticFiles

        assets = dist / "assets"
        if assets.exists():
            app.mount("/assets", StaticFiles(directory=str(assets)), name="assets")

        @app.get("/{full_path:path}", include_in_schema=False)
        def spa_fallback(full_path: str):
            """SPA 回退：非 API 路径统一返回 index.html，保证刷新子路由可用。"""
            if full_path.startswith(("api/", "docs", "openapi", "redoc")):
                raise HTTPException(status_code=404, detail="接口不存在")
            return FileResponse(dist / "index.html")

        logger.info("已挂载前端静态资源：%s", dist)

    return app


app = create_app()


def main() -> None:
    import uvicorn

    settings = global_settings
    uvicorn.run(
        "reelrank.serving.api:app",
        host=str(settings.serving.host),
        port=int(settings.serving.port),
        reload=bool(settings.serving.reload),
        log_level="info",
    )


if __name__ == "__main__":
    started = time.time()
    configure(str(global_settings.project.log_level))
    logger.info("启动 ReelRank 服务 port=%s (%.3fs)", global_settings.serving.port, time.time() - started)
    main()
