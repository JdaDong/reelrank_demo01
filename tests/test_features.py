import math
from pathlib import Path

import pandas as pd
import pytest

from reelrank.features.item_features import bayesian_quality, freshness_score
from reelrank.features.registry import cross_features

ARTIFACTS = Path("artifacts")


def test_bayesian_quality_shrinks_low_vote_items():
    vote_average = pd.Series([9.0, 7.0])
    vote_count = pd.Series([2.0, 20000.0])
    scores = bayesian_quality(vote_average, vote_count, prior_votes=200.0, prior_mean=6.5)
    assert scores.iloc[0] < 7.5          # 低票数高分被拉回先验
    assert scores.iloc[1] == pytest.approx(7.0, abs=0.01)  # 高票数保持原始分
    assert scores.iloc[1] > scores.iloc[0]


def test_freshness_decays_with_age_and_stays_in_unit_range():
    years = pd.Series([2025, 2015, 1998])
    scores = freshness_score(years, half_life_days=2555.0)
    assert scores.iloc[0] > scores.iloc[1] > scores.iloc[2]
    assert scores.between(0.0, 1.0).all()


def test_cross_features_match_user_preference():
    user = {
        "genre_affinity": {"科幻": 6.0, "剧情": 1.0},
        "keyword_affinity": {"太空": 3.0},
        "director_affinity": {"21": 2.0},
        "cast_affinity": {"11": 1.5},
        "user_era_mean": (2018 - 1990) / 40.0,
        "user_avg_rating": 4.0,
    }
    liked = {
        "genre_names": ["科幻"], "keyword_names": ["太空"], "director_ids": [21],
        "cast_ids": [11], "year": 2018, "quality_bayes": 8.0,
    }
    disliked = {
        "genre_names": ["西部"], "keyword_names": ["沙漠"], "director_ids": [99],
        "cast_ids": [88], "year": 1970, "quality_bayes": 5.0,
    }
    hit = cross_features(user, liked)
    miss = cross_features(user, disliked)

    assert hit["genre_match"] == pytest.approx(1.0)
    assert hit["director_match"] == 1.0 and miss["director_match"] == 0.0
    assert hit["keyword_match"] > miss["keyword_match"]
    assert hit["era_gap"] > miss["era_gap"] and hit["quality_gap"] > miss["quality_gap"]


@pytest.mark.skipif(not (ARTIFACTS / "item_features.parquet").exists(), reason="尚未构建特征产物")
def test_item_feature_artifacts_are_complete():
    items = pd.read_parquet(ARTIFACTS / "item_features.parquet")
    assert len(items) > 0
    assert items["quality_bayes"].between(0, 10).all()
    assert items["freshness"].between(0, 1).all()
    assert items["text"].str.len().gt(10).all()
    assert items["movie_id"].is_unique


@pytest.mark.skipif(not (ARTIFACTS / "user_features.parquet").exists(), reason="尚未构建特征产物")
def test_user_feature_artifacts_are_plausible():
    users = pd.read_parquet(ARTIFACTS / "user_features.parquet")
    assert len(users) > 0
    assert users["user_click_rate"].between(0, 1).all()
    assert users["user_activity"].gt(0).all()
    assert all(math.isfinite(x) for x in users["user_era_mean"])
    assert users["history_positive"].apply(len).max() > 0
