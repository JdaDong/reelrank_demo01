from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ARTIFACTS = Path("artifacts")

pytestmark = pytest.mark.skipif(not (ARTIFACTS / "train_meta.json").exists(), reason="尚未训练，接口依赖产物")


@pytest.fixture(scope="module")
def client():
    from reelrank.serving.api import create_app

    with TestClient(create_app()) as test_client:
        yield test_client


def test_health_reports_loaded_artifacts(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["movies"] > 0 and body["users"] > 0 and body["ads"] > 0


def test_search_returns_results_and_trace(client):
    body = client.get("/api/search", params={"query": "科幻 太空", "user_id": 3, "trace": True}).json()
    assert body["scene"] == "search"
    assert len(body["items"]) > 0
    assert [stage["stage"] for stage in body["trace"]["stages"]] == ["recall", "coarse", "fine", "rerank", "ads"]
    assert any(item["is_ad"] for item in body["items"])


def test_recommend_works_without_query(client):
    body = client.get("/api/recommend", params={"user_id": 5, "topk": 10}).json()
    assert body["scene"] == "recommend"
    organic = [item for item in body["items"] if not item["is_ad"]]
    assert len(organic) >= 9 and len(body["items"]) >= 10  # topk 指自然结果条数，广告随后插入
    assert body["trace"] is not None  # 默认按配置返回 trace（调试面板依赖）

    disabled = client.get("/api/recommend", params={"user_id": 5, "trace": False}).json()
    assert disabled["trace"] is None


def test_funnel_debug_applies_overrides(client):
    body = client.post(
        "/api/funnel/debug",
        json={"scene": "recommend", "user_id": 6, "recall_limit": 100, "coarse_topk": 40, "fine_topk": 15, "topk": 6, "ads_enabled": False},
    ).json()
    stages = {stage["stage"]: stage["output_count"] for stage in body["trace"]["stages"]}
    assert stages["coarse"] == 40 and stages["fine"] == 15 and len(body["items"]) == 6
    assert all(not item["is_ad"] for item in body["items"])


def test_funnel_debug_rejects_unknown_scene(client):
    response = client.post("/api/funnel/debug", json={"scene": "unknown"})
    assert response.status_code == 400


def test_ads_board_and_reset(client):
    client.get("/api/recommend", params={"user_id": 8})
    board = client.get("/api/ads/board").json()
    assert board["advertisers"] and board["total_budget"] > 0
    assert client.post("/api/ads/reset").json()["status"] == "ok"
    after = client.get("/api/ads/board").json()
    assert after["total_spent"] == 0.0


def test_movie_detail_and_404(client):
    movie_id = client.get("/api/recommend", params={"user_id": 2}).json()["items"][0]["movie_id"]
    detail = client.get(f"/api/movies/{movie_id}").json()
    assert detail["movie_id"] == movie_id and detail["features"] and detail["similar"]
    assert client.get("/api/movies/999999999").status_code == 404


def test_stats_exposes_funnel_config(client):
    body = client.get("/api/stats").json()
    assert body["funnel"]["coarse"]["topk"] > 0
    assert body["ads"]["slots"]
    assert "data_source" in body


def test_users_endpoint(client):
    body = client.get("/api/users", params={"limit": 5}).json()
    assert len(body["users"]) <= 5 and "top_genres" in body["users"][0]
