"""请求/响应模型：核心是贯穿四层漏斗 + 广告层的 FunnelTrace。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CandidateItem(BaseModel):
    """漏斗各层中的候选条目，携带该层已知的全部分数分量。"""

    movie_id: int
    title: str = ""
    scores: dict[str, float] = Field(default_factory=dict)
    sources: list[str] = Field(default_factory=list)
    rank: int | None = None
    rank_before: int | None = None
    cut_reason: str | None = None


class StageTrace(BaseModel):
    stage: Literal["recall", "coarse", "fine", "rerank", "ads"]
    input_count: int
    output_count: int
    latency_ms: float
    cut_reason: str
    items: list[CandidateItem] = Field(default_factory=list)
    sources: dict[str, int] = Field(default_factory=dict)


class AdItem(BaseModel):
    ad_id: int
    advertiser_id: int
    advertiser_name: str = ""
    movie_id: int
    title: str = ""
    slot: int | None = None
    pctr: float = 0.0
    bid: float = 0.0
    quality_factor: float = 1.0
    ecpm: float = 0.0
    price: float = 0.0
    blocked_reason: str | None = None


class FunnelTrace(BaseModel):
    request_id: str
    scene: Literal["search", "recommend"]
    query: str | None = None
    user_id: int | None = None
    stages: list[StageTrace] = Field(default_factory=list)
    ads: list[AdItem] = Field(default_factory=list)
    total_latency_ms: float = 0.0
    config: dict[str, float | int] = Field(default_factory=dict)


class ResultItem(BaseModel):
    movie_id: int
    title: str
    year: int = 0
    genres: list[str] = Field(default_factory=list)
    vote_average: float = 0.0
    popularity: float = 0.0
    poster_path: str = ""
    overview: str = ""
    score: float = 0.0
    pctr: float = 0.0
    sources: list[str] = Field(default_factory=list)
    is_ad: bool = False
    ad: AdItem | None = None
    rank_before: int | None = None


class RankResponse(BaseModel):
    request_id: str
    scene: str
    query: str | None = None
    user_id: int | None = None
    items: list[ResultItem] = Field(default_factory=list)
    trace: FunnelTrace | None = None
    total_latency_ms: float = 0.0
