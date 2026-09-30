"""内置样例快照：无 TMDB 凭证时生成一份 TMDB 结构的合成目录，保证端到端可跑通。

数据为确定性合成（固定随机种子），影片关系边由「类型/关键词/导演/年代」加权 Jaccard 计算，
因此 ItemCF 召回在样例数据上同样能学到真实的结构信号。
"""

from __future__ import annotations

import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path

from reelrank.config import Settings, settings as global_settings
from reelrank.logging_utils import get_logger

logger = get_logger("reelrank.data.seed")

GENRES = [
    (28, "动作"), (12, "冒险"), (16, "动画"), (35, "喜剧"), (80, "犯罪"),
    (99, "纪录片"), (18, "剧情"), (10751, "家庭"), (14, "奇幻"), (36, "历史"),
    (27, "恐怖"), (10402, "音乐"), (9648, "悬疑"), (10749, "爱情"),
    (878, "科幻"), (53, "惊悚"), (10752, "战争"), (37, "西部"),
]

KEYWORDS = [
    "太空", "人工智能", "复仇", "成长", "家庭", "战争", "赛车", "黑帮", "太空站",
    "吸血鬼", "时间旅行", "机器人", "病毒", "王室", "沙漠", "深海", "雪山",
    "孤岛", "记忆", "克隆", "魔法", "剑客", "法庭", "黑客", "病毒爆发",
    "末日", "音乐梦", "拳击", "机场", "列车", "校园", "医生", "警探",
    "间谍", "海盗", "恐龙", "外星生命", "梦境", "赌场", "监狱", "荒野生存",
    "父子", "母女", "兄弟", "友谊", "初恋", "婚姻危机", "移民", "贫民窟",
    "华尔街", "创业", "登山", "航海", "沙漠求生", "考古", "宝藏", "预言",
    "诅咒", "狼人", "丧尸", "鬼屋", "通灵", "占星", "诗歌", "绘画",
    "摄影", "舞台剧", "爵士乐", "街头赛车", "无人机", "量子", "平行宇宙",
]

TITLE_PREFIX = [
    "星际", "暗夜", "孤城", "追光", "深海", "荒原", "第七封", "无声", "长夜",
    "赤色", "铁血", "白银", "黑镜", "烈火", "静默", "逆流", "曙光", "碎裂",
    "雾中", "极地", "午夜", "黄金", "狂沙", "霜降", "回声", "风暴", "断桥",
    "归途", "迷航", "最后的", "失落的", "燃烧的", "沉默的", "透明的", "无尽的",
]

TITLE_SUFFIX = [
    "行动", "纪事", "之城", "回声", "边界", "黎明", "陷阱", "回响", "列车",
    "密码", "合约", "证人", "航线", "档案", "告白", "挽歌", "信条", "坐标",
    "残章", "博弈", "试炼", "尽头", "序曲", "标本", "遗言", "猎手", "航线图",
    "方程式", "独白", "畸变", "罗盘", "残影", "密令", "终局", "之约", "之诗",
]

SURNAMES = ["张", "李", "王", "陈", "刘", "周", "吴", "郑", "林", "何", "徐", "孙", "高", "马", "朱"]
GIVEN_NAMES = ["沐辰", "砚青", "知远", "屿川", "照野", "未晞", "南乔", "临风", "向晚", "斯年", "牧之", "霁月"]
WESTERN_GIVEN = ["Elliot", "Mara", "Silas", "Nadia", "Corvin", "Ilse", "Dorian", "Vera", "Kian", "Noor", "Rhys", "Alba"]
WESTERN_FAMILY = ["Vance", "Halloway", "Brandt", "Okafor", "Lindqvist", "Moreau", "Ashcroft", "Delgado", "Novak", "Reyes"]

COMPANIES = [
    "北岸影业", "灯塔制片", "银幕工场", "蓝鲸传媒", "拾光影业", "黑石影业",
    "远山制片", "恒星影业", "潮汐影业", "第七映画",
]

LANGUAGES = ["zh", "en", "ja", "ko", "fr", "es", "de"]

HERO = ["一名退役特工", "一位年轻的外科医生", "一名落魄的赛车手", "一位失忆的画家",
        "一名小镇警探", "一位单亲父亲", "一名天才程序员", "一位考古学者",
        "一名潜艇声呐员", "一位过气摇滚歌手"]
PLACE = ["在边境小城", "在暴雪封山的观测站", "在霓虹密布的旧港区", "在一列深夜列车上",
         "在被封锁的孤岛", "在沙漠深处的钻探营地", "在一座即将拆除的老剧院"]
CONFLICT = ["卷入一场跨国走私", "被迫追查失踪的亲人", "与自己的记忆对峙",
            "揭开十年前那场事故的真相", "卷入一桩精心设计的骗局",
            "必须在黎明前做出抉择", "与一位旧日对手重逢", "守护一份不能公开的证据"]


def _person_name(rng: random.Random) -> str:
    if rng.random() < 0.55:
        return f"{rng.choice(SURNAMES)}{rng.choice(GIVEN_NAMES)}"
    return f"{rng.choice(WESTERN_GIVEN)} {rng.choice(WESTERN_FAMILY)}"


def _title(rng: random.Random, used: set[str]) -> str:
    for _ in range(20):
        name = f"{rng.choice(TITLE_PREFIX)}{rng.choice(TITLE_SUFFIX)}"
        if name not in used:
            used.add(name)
            return name
    name = f"{rng.choice(TITLE_PREFIX)}{rng.choice(TITLE_SUFFIX)}·{len(used)}"
    used.add(name)
    return name


def _overview(rng: random.Random, genres: list[str], year: int) -> str:
    return (
        f"{year}年，一部融合{'、'.join(genres[:2])}元素的作品。"
        f"{rng.choice(HERO)}{rng.choice(PLACE)}，{rng.choice(CONFLICT)}。"
        f"随着线索层层展开，他必须面对的不仅是外部的敌人，还有自己长久以来回避的真相。"
    )


def _movie(rng: random.Random, mid: int, used_titles: set[str]) -> dict:
    n_genres = rng.choice([1, 2, 2, 3])
    genres = [{"id": gid, "name": gname} for gid, gname in rng.sample(GENRES, n_genres)]
    keywords = [{"id": 10000 + idx, "name": kw} for idx, kw in enumerate(rng.sample(KEYWORDS, rng.randint(3, 7)))]
    directors = [{"id": 200000 + rng.randint(1, 400), "name": _person_name(rng)} for _ in range(rng.choice([1, 1, 2]))]
    cast = [
        {"id": 300000 + rng.randint(1, 5000), "name": _person_name(rng), "order": i, "character": rng.choice(HERO).replace("一名", "").replace("一位", "")}
        for i in range(rng.randint(5, 10))
    ]

    popularity = float(min(900.0, max(1.0, math.exp(rng.gauss(3.1, 0.9)))))
    vote_count = int(min(120000, max(30, math.exp(rng.gauss(6.2, 1.3)))))
    vote_average = float(min(9.4, max(2.6, rng.gauss(6.3 + 0.55 * math.log10(popularity), 0.85))))
    year = int(rng.triangular(1995, 2026, 2018))
    budget = int(popularity * rng.uniform(2.0e4, 9.0e4))
    revenue = int(budget * math.exp(rng.gauss(0.9, 1.0)))

    return {
        "id": mid,
        "title": _title(rng, used_titles),
        "original_title": "",
        "overview": _overview(rng, [g["name"] for g in genres], year),
        "tagline": rng.choice(["真相从不沉默。", "这一次，没有回头路。", "在黑暗尽头，仍有光。", "选择与代价，一同到来。"]),
        "release_date": f"{year}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
        "year": year,
        "runtime": int(min(190, max(72, rng.gauss(112, 22)))),
        "budget": budget,
        "revenue": revenue,
        "popularity": round(popularity, 3),
        "vote_average": round(vote_average, 2),
        "vote_count": vote_count,
        "original_language": rng.choice(LANGUAGES),
        "adult": False,
        "status": "Released",
        "imdb_id": "",
        "poster_path": "",
        "backdrop_path": "",
        "genres": genres,
        "keywords": keywords,
        "cast": cast,
        "directors": directors,
        "companies": [{"id": 400000 + rng.randint(1, 60), "name": rng.choice(COMPANIES)} for _ in range(rng.randint(1, 3))],
    }


def _similarity(a: dict, b: dict) -> float:
    ga = {g["name"] for g in a["genres"]}
    gb = {g["name"] for g in b["genres"]}
    ka = {k["name"] for k in a["keywords"]}
    kb = {k["name"] for k in b["keywords"]}
    da = {d["name"] for d in a["directors"]}
    db = {d["name"] for d in b["directors"]}

    def jaccard(x: set, y: set) -> float:
        if not x or not y:
            return 0.0
        return len(x & y) / len(x | y)

    era_gap = abs((a["year"] or 2000) - (b["year"] or 2000))
    era = math.exp(-era_gap / 12.0)
    return 0.42 * jaccard(ga, gb) + 0.33 * jaccard(ka, kb) + 0.15 * jaccard(da, db) + 0.10 * era


def _build_edges(rng: random.Random, movies: list[dict], topk: int = 12, sample: int = 240) -> list[dict]:
    total = len(movies)
    sample = min(sample, total)
    edges: list[dict] = []
    for movie in movies:
        candidates = rng.sample(movies, sample)
        scored = sorted(
            ((_similarity(movie, other), other["id"]) for other in candidates if other["id"] != movie["id"]),
            key=lambda x: x[0],
            reverse=True,
        )[:topk]
        similar = [mid for score, mid in scored if score > 0.14]
        cut = max(1, len(similar) // 2)
        edges.append({"id": movie["id"], "similar": similar, "recommended": similar[:cut]})
    return edges


def build_seed_snapshot(settings: Settings | None = None, size: int = 600, force: bool = False) -> dict:
    """生成并落盘样例快照（已存在且非 force 时直接复用）。"""
    settings = settings or global_settings
    movies_path: Path = settings.path("data/seed/movies.json")
    edges_path: Path = settings.path("data/seed/similar.json")
    if movies_path.exists() and edges_path.exists() and not force:
        logger.info("复用已有样例快照：%s", movies_path)
        movies = json.loads(movies_path.read_text(encoding="utf-8"))
        edges = json.loads(edges_path.read_text(encoding="utf-8"))
        return {"source": "seed_snapshot", "movie_count": len(movies), "edge_count": len(edges), "reused": True}

    rng = random.Random(int(settings.behavior.synthetic.seed))
    used_titles: set[str] = set()
    movies = [_movie(rng, 500000 + i, used_titles) for i in range(size)]
    edges = _build_edges(rng, movies)

    movies_path.write_text(json.dumps(movies, ensure_ascii=False), encoding="utf-8")
    edges_path.write_text(json.dumps(edges, ensure_ascii=False), encoding="utf-8")
    logger.info("生成样例快照：%s 部影片 → %s", len(movies), movies_path)
    return {"source": "seed_snapshot", "movie_count": len(movies), "edge_count": len(edges), "reused": False}


def materialize_seed_to_raw(settings: Settings | None = None) -> dict:
    """把样例快照写入 data/raw，使下游 ETL 无需关心数据来源。"""
    settings = settings or global_settings
    meta = build_seed_snapshot(settings)
    seed_movies = settings.path("data/seed/movies.json")
    seed_edges = settings.path("data/seed/similar.json")

    movies_path = settings.path("data/raw/movies.jsonl")
    edges_path = settings.path("data/raw/similar.jsonl")
    movies_path.write_text(
        "\n".join(json.dumps(m, ensure_ascii=False) for m in json.loads(seed_movies.read_text(encoding="utf-8"))) + "\n",
        encoding="utf-8",
    )
    edges_path.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in json.loads(seed_edges.read_text(encoding="utf-8"))) + "\n",
        encoding="utf-8",
    )

    meta_path = settings.path("data/raw/meta.json")
    payload = {
        "source": "seed_snapshot",
        "movie_count": meta["movie_count"],
        "edge_count": meta["edge_count"],
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "note": "无 TMDB 凭证，使用内置确定性合成样例快照",
    }
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.warning("未检测到 TMDB 凭证，已使用内置样例快照兜底（%s 部）", meta["movie_count"])
    return payload


if __name__ == "__main__":
    print(materialize_seed_to_raw())
