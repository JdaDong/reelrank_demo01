"""自然结果与广告的混排：固定广告位插入 + 同源去重。"""

from __future__ import annotations

from dataclasses import dataclass

from reelrank.recall.multi_recall import Candidate
from reelrank.serving.schemas import AdItem


@dataclass
class MixedRow:
    movie_id: int
    candidate: Candidate | None = None
    ad: AdItem | None = None

    @property
    def is_ad(self) -> bool:
        return self.ad is not None


def mix_results(organic: list[Candidate], ads: list[AdItem], slots: list[int]) -> tuple[list[MixedRow], list[int]]:
    """把中标广告插入自然结果的指定位次，返回 (混排结果, 被广告替换掉的自然结果 movie_id)。"""
    ad_movie_ids = {int(ad.movie_id) for ad in ads}
    kept = [c for c in organic if int(c.movie_id) not in ad_movie_ids]
    removed = [int(c.movie_id) for c in organic if int(c.movie_id) in ad_movie_ids]

    rows: list[MixedRow] = [MixedRow(movie_id=int(c.movie_id), candidate=c) for c in kept]
    for position, ad in enumerate(ads):
        slot = int(ad.slot if ad.slot is not None else slots[position] if position < len(slots) else len(rows))
        row = MixedRow(movie_id=int(ad.movie_id), ad=ad)
        if slot < len(rows):
            rows.insert(slot, row)
        else:
            rows.append(row)
            ad.slot = len(rows) - 1
    return rows, removed
