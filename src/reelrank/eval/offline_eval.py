"""离线评估报告：召回/排序/广告/耗时全链路指标。

- 推荐场景：以用户训练段历史为 trigger，评估各路召回 Recall@K 与最终结果的 NDCG@K
- 搜索场景：合成 query，评估 BM25/向量召回与最终排序的相关性
- 排序模型：粗排/精排 AUC 与 GAUC（按请求加权）
- 广告与耗时：多次真实请求的 eCPM、二价扣费、RPM 与各层耗时均值/P95
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from reelrank.config import Settings, settings as global_settings
from reelrank.eval.metrics import auc, gauc, ndcg_at_k, percentile, precision_at_k, recall_at_k, rpm
from reelrank.features.item_features import build_item_features
from reelrank.features.registry import load_warehouse
from reelrank.features.user_features import build_user_features
from reelrank.logging_utils import get_logger
from reelrank.rank.coarse import run_coarse
from reelrank.rank.fine import run_fine
from reelrank.rank.rerank import run_rerank
from reelrank.recall.multi_recall import MultiRecall
from reelrank.serving.funnel import run_funnel
from reelrank.serving.state import RankState, get_state
from reelrank.train.sampler import build_user_profiles, split_history_frames

logger = get_logger("reelrank.eval.offline")


def _similar_map(settings: Settings) -> dict[int, list[int]]:
    edges = load_warehouse(settings, tables=["item_sim_edge"]).get("item_sim_edge", pd.DataFrame())
    mapping: dict[int, list[int]] = {}
    if not edges.empty:
        sim = edges[edges["relation"].isin(["similar", "recommendation", "recommended"])]
        for src, group in sim.groupby("src_movie_id"):
            mapping[int(src)] = [int(x) for x in group.sort_values("rank_pos")["dst_movie_id"].tolist()]
    return mapping


def _cascade(state: RankState, candidates, profile: dict, settings: Settings) -> tuple[list[int], dict[str, float]]:
    """跑完粗排 → 精排 → 重排，返回最终 movie_id 列表与各层耗时。"""
    timings: dict[str, float] = {}
    kept, coarse_trace = run_coarse(state, candidates, profile, settings)
    timings["coarse"] = coarse_trace.latency_ms
    kept, fine_trace = run_fine(state, kept, profile, settings)
    timings["fine"] = fine_trace.latency_ms
    kept, rerank_trace = run_rerank(state, kept, settings)
    timings["rerank"] = rerank_trace.latency_ms
    return [int(c.movie_id) for c in kept], timings


def evaluate_recommend(state: RankState, settings: Settings, history: pd.DataFrame, sample_users: int, topk_route: int, ks: list[int]) -> dict:
    train, test = split_history_frames(history, float(settings.eval.test_ratio))
    train_pos = {
        int(uid): [int(x) for x in group[group["label"] >= 1]["movie_id"].tolist()]
        for uid, group in train.groupby("user_id", sort=False)
    }
    train_all = {int(uid): [int(x) for x in group["movie_id"].tolist()] for uid, group in train.groupby("user_id", sort=False)}
    test_pos = {
        int(uid): {int(x) for x in group[group["label"] >= 1]["movie_id"].tolist()}
        for uid, group in test.groupby("user_id", sort=False)
    }

    eligible = [u for u in test_pos if len(train_pos.get(u, [])) >= 5 and len(test_pos[u]) >= 1]
    rng = random.Random(int(settings.eval.seed))
    sampled = rng.sample(eligible, min(sample_users, len(eligible))) if eligible else []
    if not sampled:
        return {"users": 0}

    recall_engine = MultiRecall(state, settings)
    routes: dict[str, list[float]] = {"itemcf": [], "vector": [], "hot": [], "merged": []}
    ndcg: dict[int, list[float]] = {k: [] for k in ks}
    precision5: list[float] = []
    final_recall: dict[int, list[float]] = {k: [] for k in ks}

    items = state.items
    for user_id in sampled:
        positives = train_pos[user_id]
        exclude = set(train_all[user_id])
        relevant = test_pos[user_id]

        # 训练段画像（避免用全量历史导致标签泄漏）
        user_train = train[train["user_id"] == user_id]
        profiles = build_user_profiles(build_user_features(settings, items, persist=False, history=user_train))
        profile = profiles.get(user_id) or state.default_profile

        triggers = [(mid, 1.0 - 0.4 * i / max(1, len(positives))) for i, mid in enumerate(positives[-20:][::-1])]
        user_vector = state.vector.user_vector([(mid, 1.0) for mid in positives[-30:]])

        route_ids = {
            "itemcf": [int(m) for m, _ in state.itemcf.recall(triggers, topk=topk_route, exclude=exclude)],
            "vector": [int(m) for m, _ in state.vector.search(user_vector, topk=topk_route, exclude=exclude)],
            "hot": [int(m) for m, _ in state.hot_ranking()[:topk_route]],
        }
        merged = list(dict.fromkeys(route_ids["itemcf"] + route_ids["vector"] + route_ids["hot"]))
        route_ids["merged"] = merged

        for name, ids in route_ids.items():
            routes[name].append(recall_at_k(ids, relevant, topk_route))

        candidates, _ = recall_engine.recall(query=None, user_id=user_id, history=positives)
        final_ids, _ = _cascade(state, candidates, profile, settings)
        for k in ks:
            ndcg[k].append(ndcg_at_k(final_ids, relevant, k))
            final_recall[k].append(recall_at_k(final_ids, relevant, k))
        precision5.append(precision_at_k(final_ids, relevant, 5))

    return {
        "users": len(sampled),
        "route_recall": {name: round(sum(values) / len(values), 4) for name, values in routes.items() if values},
        "final_ndcg": {f"ndcg@{k}": round(sum(ndcg[k]) / len(ndcg[k]), 4) for k in ks if ndcg[k]},
        "final_recall": {f"recall@{k}": round(sum(final_recall[k]) / len(final_recall[k]), 4) for k in ks if final_recall[k]},
        "precision@5": round(sum(precision5) / len(precision5), 4) if precision5 else 0.0,
    }


def evaluate_search(state: RankState, settings: Settings, sample_queries: int, topk_route: int, ks: list[int]) -> dict:
    similar = _similar_map(settings)
    items = state.items
    rng = random.Random(int(settings.eval.seed))
    pool = items.sample(n=min(sample_queries, len(items)), random_state=int(settings.eval.seed))
    recall_engine = MultiRecall(state, settings)

    routes: dict[str, list[float]] = {"bm25": [], "vector": [], "hot": [], "merged": []}
    ndcg: dict[int, list[float]] = {k: [] for k in ks}
    precision3: list[float] = []
    profile = state.default_profile

    for row in pool.to_dict("records"):
        movie_id = int(row["movie_id"])
        keywords = list(row["keyword_names"])[:2]
        query = " ".join([str(row["title"]), *keywords]).strip()
        relevant = {movie_id} | {int(m) for m in similar.get(movie_id, [])[:10]}

        bm25_ids = [int(m) for m, _ in state.bm25.search(query, topk=topk_route)]
        query_vector = state.vector.query_vector(query, state.bm25, topn=20)
        vector_ids = [int(m) for m, _ in state.vector.search(query_vector, topk=topk_route)]
        hot_ids = [int(m) for m, _ in state.hot_ranking()[:topk_route]]
        route_ids = {"bm25": bm25_ids, "vector": vector_ids, "hot": hot_ids,
                     "merged": list(dict.fromkeys(bm25_ids + vector_ids + hot_ids))}

        for name, ids in route_ids.items():
            routes[name].append(recall_at_k(ids, relevant, topk_route))

        candidates, _ = recall_engine.recall(query=query)
        final_ids, _ = _cascade(state, candidates, profile, settings)
        for k in ks:
            ndcg[k].append(ndcg_at_k(final_ids, relevant, k))
        precision3.append(precision_at_k(final_ids, relevant, 3))

    return {
        "queries": len(pool),
        "route_recall": {name: round(sum(values) / len(values), 4) for name, values in routes.items() if values},
        "final_ndcg": {f"ndcg@{k}": round(sum(ndcg[k]) / len(ndcg[k]), 4) for k in ks if ndcg[k]},
        "precision@3": round(sum(precision3) / len(precision3), 4) if precision3 else 0.0,
        "note": "搜索标签为相关性代理（query 源影片 + TMDB 相似邻居）",
    }


def evaluate_rankers(settings: Settings) -> dict:
    path = settings.path("artifacts/train_samples.parquet")
    if not path.exists():
        return {}
    frame = pd.read_parquet(path)
    fine_topk = int(settings.train.fine_candidates_per_sample)
    fine_frame = (
        frame.sort_values(["sample_id", "coarse_score"], ascending=[True, False])
        .groupby("sample_id", group_keys=False)
        .head(fine_topk)
    )
    by_scene = {
        str(scene): round(auc(group["label"], group["coarse_score"]), 4)
        for scene, group in frame.groupby("scene")
    }
    return {
        "coarse_auc": round(auc(frame["label"], frame["coarse_score"]), 4),
        "coarse_auc_by_scene": by_scene,
        "fine_auc": round(auc(fine_frame["label"], fine_frame["pctr"]), 4),
        "fine_gauc": round(gauc(fine_frame, "sample_id", "label", "pctr"), 4),
        "samples": int(frame["sample_id"].nunique()),
    }


def evaluate_ads_and_latency(state: RankState, settings: Settings, requests: int) -> tuple[dict, dict]:
    rng = random.Random(int(settings.eval.seed))
    user_ids = sorted(state.profiles.keys())[:500]
    titles = [str(t) for t in state.items["title"].head(200).tolist()]

    latencies: dict[str, list[float]] = {"recall": [], "coarse": [], "fine": [], "rerank": [], "ads": []}
    totals: list[float] = []
    impressions = 0
    ecpm_sum = 0.0
    price_sum = 0.0
    spend_before = sum(state.ad_index.spent.values())

    for index in range(requests):
        scene = "search" if index % 2 == 0 else "recommend"
        query = rng.choice(titles) if scene == "search" else None
        response = run_funnel(
            state,
            scene=scene,
            query=query,
            user_id=rng.choice(user_ids) if user_ids else None,
            session_id=f"eval-{index}",
            with_trace=True,
            settings=settings,
        )
        totals.append(response.total_latency_ms)
        for stage in response.trace.stages if response.trace else []:
            latencies.setdefault(stage.stage, []).append(stage.latency_ms)
        for ad in (response.trace.ads if response.trace else []):
            if ad.blocked_reason is None:
                impressions += 1
                ecpm_sum += ad.ecpm
                price_sum += ad.price

    spend_after = sum(state.ad_index.spent.values())
    revenue = spend_after - spend_before
    ads_metrics = {
        "requests": requests,
        "impressions": impressions,
        "avg_ecpm": round(ecpm_sum / impressions, 2) if impressions else 0.0,
        "avg_price": round(price_sum / impressions, 4) if impressions else 0.0,
        "revenue": round(revenue, 4),
        "rpm": round(rpm(revenue, impressions), 2) if impressions else 0.0,
    }
    latency_metrics = {
        stage: {"avg_ms": round(sum(values) / len(values), 2), "p95_ms": round(percentile(values, 95), 2)}
        for stage, values in latencies.items()
        if values
    }
    latency_metrics["total"] = {
        "avg_ms": round(sum(totals) / len(totals), 2) if totals else 0.0,
        "p95_ms": round(percentile(totals, 95), 2),
    }
    return ads_metrics, latency_metrics


def _markdown(report: dict) -> str:
    def table(title: str, mapping: dict) -> str:
        if not mapping:
            return f"### {title}\n\n暂无数据\n"
        rows = "\n".join(f"| {key} | {value} |" for key, value in mapping.items())
        return f"### {title}\n\n| 指标 | 值 |\n| --- | --- |\n{rows}\n"

    recommend = report.get("recommend", {})
    search = report.get("search", {})
    return "\n".join(
        [
            "# ReelRank 离线评估报告",
            "",
            f"- 生成时间：{report.get('generated_at')}",
            f"- 数据来源：{report.get('data_source')} / 行为来源：{report.get('behavior_source')}",
            "",
            "## 推荐场景",
            "",
            table("各路召回 Recall@K（训练段历史触发，测试段正例为标签）", recommend.get("route_recall", {})),
            table(
                "最终排序结果（对照随机基线）",
                {
                    **recommend.get("final_ndcg", {}),
                    **recommend.get("final_recall", {}),
                    "precision@5": recommend.get("precision@5"),
                    **{f"baseline_{key}": value for key, value in (recommend.get("random_baseline") or {}).items()},
                },
            ),
            "## 搜索场景",
            "",
            table("各路召回 Recall@K", search.get("route_recall", {})),
            table("最终排序结果", {**search.get("final_ndcg", {}), "precision@3": search.get("precision@3")}),
            "## 排序模型",
            "",
            table("粗排 / 精排", report.get("ranking", {})),
            "## 广告",
            "",
            table("竞价与收入", report.get("ads", {})),
            "## 漏斗耗时（ms）",
            "",
            table("各层耗时", {k: f"avg {v['avg_ms']} / p95 {v['p95_ms']}" for k, v in report.get("latency_ms", {}).items()}),
        ]
    )


def run_evaluation(settings: Settings | None = None, sample_users: int = 120, sample_queries: int = 120, requests: int = 20, topk_route: int = 100) -> dict:
    settings = settings or global_settings
    ks = [int(k) for k in settings.eval.ks]
    state = get_state(settings)

    history_path = settings.path("artifacts/user_history.parquet")
    if not history_path.exists():
        raise RuntimeError("缺少 artifacts/user_history.parquet，请先运行特征流水线")
    history = pd.read_parquet(history_path)

    logger.info("开始评估：%s 个用户 / %s 条 query / %s 次在线请求", sample_users, sample_queries, requests)
    recommend = evaluate_recommend(state, settings, history, sample_users, topk_route, ks)
    search = evaluate_search(state, settings, sample_queries, topk_route, ks)

    # 随机基线：Recall@K ≈ K / 候选池大小，用于判断排序是否有真实增益
    pool_size = max(1, len(state.item_raw))
    baseline = {f"recall@{k}": round(min(1.0, k / pool_size), 4) for k in ks}
    recommend["random_baseline"] = baseline
    search["random_baseline"] = baseline
    ranking = evaluate_rankers(settings)
    ads, latency = evaluate_ads_and_latency(state, settings, requests)

    warehouse_meta_path = settings.path("data/warehouse/meta.json")
    warehouse_meta = json.loads(warehouse_meta_path.read_text(encoding="utf-8")) if warehouse_meta_path.exists() else {}
    behavior_meta_path = settings.path("data/raw/behavior_meta.json")
    behavior_meta = json.loads(behavior_meta_path.read_text(encoding="utf-8")) if behavior_meta_path.exists() else {}

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_source": warehouse_meta.get("source"),
        "behavior_source": behavior_meta.get("source"),
        "recommend": recommend,
        "search": search,
        "ranking": ranking,
        "ads": ads,
        "latency_ms": latency,
    }

    json_path = settings.path("artifacts/eval_report.json")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path = settings.path("reports/eval_report.md")
    md_path.write_text(_markdown(report), encoding="utf-8")

    logger.info(
        "评估完成：推荐 NDCG@10=%.4f | 搜索 NDCG@10=%.4f | 精排 AUC=%.4f | 广告 eCPM=%.1f | P95=%.1fms",
        recommend.get("final_ndcg", {}).get("ndcg@10", 0.0),
        search.get("final_ndcg", {}).get("ndcg@10", 0.0),
        ranking.get("fine_auc", 0.0),
        ads.get("avg_ecpm", 0.0),
        latency.get("total", {}).get("p95_ms", 0.0),
    )
    logger.info("报告已生成：%s / %s", json_path, md_path)
    return report


if __name__ == "__main__":
    print(json.dumps(run_evaluation(), ensure_ascii=False, indent=2))
