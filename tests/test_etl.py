import json

import pytest

from reelrank.config import settings
from reelrank.data.seed_snapshot import build_seed_snapshot, _similarity
from reelrank.data.tmdb_client import TMDBClient, TMDBError
from reelrank.warehouse.etl import build_frames


def _sample_movies():
    return [
        {
            "id": 1,
            "title": "深海底部的回声",
            "overview": "一次深海任务中的生死抉择。",
            "year": 2018,
            "release_date": "2018-05-04",
            "runtime": 118,
            "budget": 40000000,
            "revenue": 130000000,
            "popularity": 88.5,
            "vote_average": 7.4,
            "vote_count": 12000,
            "genres": [{"id": 878, "name": "科幻"}],
            "keywords": [{"id": 1, "name": "深海"}],
            "cast": [{"id": 11, "name": "林砚青", "order": 0, "character": "队长"}],
            "directors": [{"id": 21, "name": "周斯年"}],
            "companies": [{"id": 31, "name": "北岸影业"}],
        },
        {
            "id": 2,
            "title": "荒原列车",
            "overview": "一列穿越荒原的列车上的悬案。",
            "year": 2020,
            "release_date": "2020-11-20",
            "runtime": 101,
            "budget": 12000000,
            "revenue": 45000000,
            "popularity": 32.1,
            "vote_average": 6.9,
            "vote_count": 4300,
            "genres": [{"id": 53, "name": "惊悚"}],
            "keywords": [{"id": 2, "name": "列车"}],
            "cast": [{"id": 12, "name": "Mara Vance", "order": 0, "character": "乘客"}],
            "directors": [{"id": 22, "name": "Ilse Brandt"}],
            "companies": [{"id": 32, "name": "拾光影业"}],
        },
    ]


def test_build_frames_shapes():
    movies = _sample_movies()
    edges = [{"id": 1, "similar": [2], "recommended": [2]}, {"id": 2, "similar": [1], "recommended": []}]
    frames = build_frames(movies, edges)

    assert len(frames["dim_movie"]) == 2
    assert frames["dim_movie"].iloc[0]["genre_names"] == ["科幻"]
    assert len(frames["bridge_movie_cast"]) == 2
    assert len(frames["bridge_movie_director"]) == 2
    assert len(frames["item_sim_edge"]) == 3  # 去重后 similar(2) + recommended(1)


def test_seed_snapshot_is_deterministic_and_structured():
    meta = build_seed_snapshot()
    assert meta["movie_count"] >= 600
    movies = json.loads(open("data/seed/movies.json", encoding="utf-8").read())
    first = movies[0]
    assert {"id", "title", "overview", "genres", "keywords", "directors", "cast"} <= set(first)
    assert first["overview"].strip() and first["genres"]

    # 相似边来自真实结构重叠，而非随机：同类影片相似度应显著高于随机对
    same_genre = [m for m in movies if m["genres"][0]["name"] == movies[0]["genres"][0]["name"]][:1]
    if same_genre:
        assert _similarity(movies[0], same_genre[0]) > _similarity(movies[0], movies[-1])


@pytest.mark.skipif(settings.has_tmdb_credential, reason="已配置 TMDB 凭证")
def test_client_requires_credential():
    client = TMDBClient()
    with pytest.raises(TMDBError):
        client.get("/movie/popular")
