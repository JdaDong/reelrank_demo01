from pathlib import Path

import pytest

from reelrank.ads.bidding import compute_ecpm, gsp_price, quality_factor
from reelrank.ads.mixer import mix_results
from reelrank.config import settings
from reelrank.recall.multi_recall import Candidate
from reelrank.serving.schemas import AdItem

ARTIFACTS = Path("artifacts")


def test_ecpm_is_monotonic_in_pctr_and_bid():
    base = compute_ecpm(0.05, 2.0, 1.0, 1000.0)
    assert compute_ecpm(0.10, 2.0, 1.0, 1000.0) > base
    assert compute_ecpm(0.05, 3.0, 1.0, 1000.0) > base
    assert base == pytest.approx(100.0)  # 0.05 × 2 × 1 × 1000


def test_gsp_price_respects_bid_and_reserve():
    settings_ads = settings.ads
    reserve_cpc = float(settings_ads.reserve_price) / float(settings_ads.ecpm_scale)
    price = gsp_price(next_ecpm=80.0, pctr=0.05, quality=1.0, bid=2.0, settings=settings)
    assert reserve_cpc <= price <= 2.0

    cheaper = gsp_price(next_ecpm=40.0, pctr=0.05, quality=1.0, bid=2.0, settings=settings)
    assert cheaper < price  # 下一名更弱 → 扣费更低


def test_quality_factor_is_clamped():
    low = quality_factor(settings, vote_average=2.0, vote_count=10)
    high = quality_factor(settings, vote_average=9.5, vote_count=50000)
    assert low < high
    assert float(settings.ads.quality_factor_min) <= low <= high <= float(settings.ads.quality_factor_max)


def _ad(ad_id: int, movie_id: int, slot: int) -> AdItem:
    return AdItem(ad_id=ad_id, advertiser_id=1000 + ad_id, advertiser_name=f"广告主{ad_id}",
                  movie_id=movie_id, title=f"广告{movie_id}", slot=slot, ecpm=100.0)


def test_mixer_inserts_ads_at_slots_and_dedupes():
    organic = [Candidate(movie_id=i, scores={"fine_score": 1.0 - i / 10}) for i in range(1, 11)]
    ads = [_ad(1, 5, 3), _ad(2, 99, 8)]
    rows, removed = mix_results(organic, ads, [3, 8])

    assert removed == [5], "与自然结果重复的广告素材应剔除自然结果"
    assert [i for i, row in enumerate(rows) if row.is_ad] == [3, 8]
    assert all(row.movie_id for row in rows)


def test_mixer_appends_when_slot_beyond_list():
    organic = [Candidate(movie_id=1), Candidate(movie_id=2)]
    rows, _ = mix_results(organic, [_ad(1, 77, 9)], [9])
    assert rows[-1].is_ad and rows[-1].movie_id == 77


@pytest.mark.skipif(not (ARTIFACTS / "train_meta.json").exists(), reason="尚未训练")
def test_funnel_stages_shrink_monotonically():
    from reelrank.serving.funnel import run_funnel
    from reelrank.serving.state import get_state

    state = get_state()
    response = run_funnel(state, scene="search", query="科幻 太空", user_id=3, session_id="test-monotonic")
    stages = response.trace.stages
    assert [s.stage for s in stages] == ["recall", "coarse", "fine", "rerank", "ads"]
    counts = [s.output_count for s in stages[:4]]
    assert counts == sorted(counts, reverse=True), "漏斗候选量应逐层收敛"
    assert len(response.items) >= stages[3].output_count
    assert any(item.is_ad for item in response.items)


@pytest.mark.skipif(not (ARTIFACTS / "train_meta.json").exists(), reason="尚未训练")
def test_frequency_cap_limits_same_advertiser():
    from reelrank.serving.funnel import run_funnel
    from reelrank.serving.state import get_state

    state = get_state()
    session = "test-freq"
    wins: dict[int, int] = {}
    for _ in range(3):
        response = run_funnel(state, scene="recommend", user_id=11, session_id=session)
        for ad in response.trace.ads:
            if ad.blocked_reason is None:
                wins[ad.advertiser_id] = wins.get(ad.advertiser_id, 0) + 1
    assert wins, "应产生中标广告"
    assert max(wins.values()) <= int(settings.ads.freq_cap_per_session)


@pytest.mark.skipif(not (ARTIFACTS / "train_meta.json").exists(), reason="尚未训练")
def test_ads_can_be_disabled():
    from reelrank.serving.funnel import run_funnel
    from reelrank.serving.state import get_state

    state = get_state()
    response = run_funnel(state, scene="recommend", user_id=12, session_id="test-noads", ads_enabled=False)
    assert all(not item.is_ad for item in response.items)
    assert [s.stage for s in response.trace.stages] == ["recall", "coarse", "fine", "rerank"]


@pytest.mark.skipif(not (ARTIFACTS / "train_meta.json").exists(), reason="尚未训练")
def test_debug_overrides_change_stage_sizes():
    from reelrank.serving.funnel import run_funnel
    from reelrank.serving.state import get_state

    state = get_state()
    response = run_funnel(
        state, scene="recommend", user_id=13, session_id="test-override",
        recall_limit=120, coarse_topk=60, fine_topk=20, topk=8, ads_enabled=False,
    )
    stages = {s.stage: s.output_count for s in response.trace.stages}
    assert stages["recall"] <= 120 and stages["coarse"] == 60 and stages["fine"] == 20
    assert len(response.items) == 8
