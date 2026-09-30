import { useState } from "react";
import { BadgeCheck, Star, TrendingUp } from "lucide-react";
import type { ResultItem } from "../api/client";
import { posterUrl } from "../api/client";

interface Props {
  item: ResultItem;
  index: number;
}

const SOURCE_LABEL: Record<string, string> = {
  bm25: "文本召回",
  vector: "向量召回",
  itemcf: "ItemCF",
  hot: "热门",
  ads: "广告",
  explore: "探索",
};

export default function MovieCard({ item, index }: Props) {
  const [hovered, setHovered] = useState(false);
  const poster = posterUrl(item.poster_path);
  const score = item.is_ad ? (item.ad?.ecpm ?? 0) / 200 : item.score;

  return (
    <article
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      className={`glass glass-hover relative animate-fadeUp overflow-hidden ${item.is_ad ? "border-gold/45" : ""}`}
      style={{ animationDelay: `${Math.min(index * 45, 600)}ms` }}
    >
      <div className="relative h-[220px] overflow-hidden bg-gradient-to-br from-ink-700 via-ink-800 to-ink-900">
        {poster ? (
          <img src={poster} alt={item.title} loading="lazy" className="h-full w-full object-cover opacity-90" />
        ) : (
          <div className="flex h-full w-full items-center justify-center bg-[radial-gradient(circle_at_30%_20%,rgba(255,176,32,0.22),transparent_60%)]">
            <span className="px-6 text-center text-[28px] font-bold leading-snug text-mist-100/85">{item.title}</span>
          </div>
        )}
        <div className="absolute inset-x-0 bottom-0 h-24 bg-gradient-to-t from-ink-900 to-transparent" />
        <div className="absolute left-3 top-3 flex items-center gap-2">
          <span className="rounded-lg bg-ink-900/80 px-2 py-1 text-[11px] font-semibold text-gold">#{index + 1}</span>
          {item.is_ad ? (
            <span className="flex items-center gap-1 rounded-lg bg-gradient-to-r from-gold to-ember px-2 py-1 text-[11px] font-bold text-ink-900">
              <BadgeCheck size={12} /> 广告
            </span>
          ) : null}
        </div>
        {item.rank_before !== null && item.rank_before !== index ? (
          <span className="absolute right-3 top-3 rounded-lg bg-ink-900/80 px-2 py-1 text-[11px] text-mist-300">
            重排前 #{item.rank_before + 1}
          </span>
        ) : null}
      </div>

      <div className="space-y-2.5 p-4">
        <div className="flex items-start justify-between gap-3">
          <h3 className="line-clamp-1 text-[15px] font-semibold text-mist-100">{item.title}</h3>
          <span className="flex shrink-0 items-center gap-1 text-[12px] text-gold">
            <Star size={12} fill="currentColor" />
            {item.vote_average.toFixed(1)}
          </span>
        </div>
        <div className="flex flex-wrap items-center gap-1.5 text-[11px] text-mist-500">
          <span>{item.year}</span>
          {item.genres.slice(0, 3).map((genre) => (
            <span key={genre} className="chip">
              {genre}
            </span>
          ))}
        </div>

        <div className="space-y-1.5">
          <div className="flex items-center justify-between text-[11px] text-mist-500">
            <span>{item.is_ad ? "eCPM 竞争力" : "精排分"}</span>
            <span className="text-mist-300">{score.toFixed(3)}</span>
          </div>
          <div className="h-1.5 overflow-hidden rounded-full bg-white/8">
            <div
              className={`h-full rounded-full transition-all duration-700 ${item.is_ad ? "bg-gradient-to-r from-gold to-ember" : "bg-gradient-to-r from-aqua to-gold"}`}
              style={{ width: `${Math.min(100, Math.max(4, score * 100))}%` }}
            />
          </div>
          {item.is_ad && item.ad ? (
            <div className="flex items-center gap-2 text-[11px] text-mist-400">
              <TrendingUp size={12} className="text-gold" />
              {item.ad.advertiser_name} · pCTR {(item.ad.pctr * 100).toFixed(2)}% · 出价 ¥{item.ad.bid} · 二价 ¥{item.ad.price}
            </div>
          ) : (
            <div className="flex items-center gap-2 text-[11px] text-mist-400">
              <span>pCTR {(item.pctr * 100).toFixed(2)}%</span>
              <span className="text-mist-500">来源：{(item.sources || []).map((s) => SOURCE_LABEL[s] || s).join(" / ") || "-"}</span>
            </div>
          )}
        </div>

        <p className="line-clamp-2 text-[12px] leading-relaxed text-mist-500">{item.overview || "暂无简介"}</p>
      </div>

      {hovered && !item.is_ad ? (
        <div className="pointer-events-none absolute inset-x-0 bottom-0 border-t border-white/10 bg-ink-900/95 p-4 text-[11px] text-mist-300 backdrop-blur-xl">
          <div className="mb-2 font-semibold text-gold">为什么推荐给我</div>
          <div className="flex flex-wrap gap-1.5">
            {(item.sources || []).map((source) => (
              <span key={source} className="chip">
                {SOURCE_LABEL[source] || source}
              </span>
            ))}
          </div>
          <div className="mt-2 text-mist-500">精排分由 pCTR × 0.7 + 质量 × 0.2 + 新鲜度 × 0.1 融合得到</div>
        </div>
      ) : null}
    </article>
  );
}
