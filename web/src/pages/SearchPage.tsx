import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { Clock3, Gauge, Search, Sparkles, Zap } from "lucide-react";
import { api, type RankResponse } from "../api/client";
import MovieCard from "../components/MovieCard";
import FunnelWaterfall from "../components/FunnelWaterfall";

const HOT_QUERIES = ["科幻 太空", "动作 追车", "悬疑 真相", "动画 家庭", "爱情 城市", "恐怖 孤岛", "纪录片 自然", "战争 人性"];

interface Props {
  userId: number | null;
}

export default function SearchPage({ userId }: Props) {
  const [params, setParams] = useSearchParams();
  const query = params.get("q") || "";
  const [topk, setTopk] = useState(20);
  const [adsOn, setAdsOn] = useState(true);
  const [data, setData] = useState<RankResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!query) {
      setData(null);
      return;
    }
    setLoading(true);
    setError(null);
    api
      .search({ query, user_id: userId, topk, ads: adsOn, session_id: "web-search" })
      .then(setData)
      .catch((err: Error) => setError(err.message))
      .finally(() => setLoading(false));
  }, [query, userId, topk, adsOn]);

  const stages = data?.trace?.stages ?? [];
  const adCount = useMemo(() => (data?.items || []).filter((item) => item.is_ad).length, [data]);

  return (
    <div className="grid gap-6 xl:grid-cols-[250px_minmax(0,1fr)_330px]">
      <aside className="space-y-4">
        <div className="glass p-4">
          <div className="mb-3 text-[13px] font-semibold text-mist-100">热门检索词</div>
          <div className="flex flex-wrap gap-2">
            {HOT_QUERIES.map((word) => (
              <button
                key={word}
                type="button"
                onClick={() => setParams({ q: word })}
                className={`chip transition-colors hover:border-gold/50 hover:text-gold ${query === word ? "border-gold/60 text-gold" : ""}`}
              >
                {word}
              </button>
            ))}
          </div>
        </div>

        <div className="glass space-y-4 p-4">
          <div className="text-[13px] font-semibold text-mist-100">请求参数</div>
          <label className="block space-y-2">
            <span className="text-[12px] text-mist-500">结果条数 topk：{topk}</span>
            <input type="range" min={6} max={40} value={topk} onChange={(event) => setTopk(Number(event.target.value))} className="w-full accent-gold" />
          </label>
          <label className="flex items-center justify-between text-[12px] text-mist-300">
            投放广告
            <input type="checkbox" checked={adsOn} onChange={(event) => setAdsOn(event.target.checked)} className="h-4 w-4 accent-gold" />
          </label>
        </div>
      </aside>

      <section className="space-y-4">
        <div className="glass flex items-center justify-between px-5 py-4">
          <div>
            <div className="text-[15px] font-semibold text-mist-100">
              {query ? `“${query}” 的排序结果` : "输入关键词开始检索"}
            </div>
            <div className="mt-1 text-[12px] text-mist-500">
              {data ? `共 ${data.items.length} 条（含 ${adCount} 条广告）· request ${data.request_id}` : "走完整四层漏斗：召回 → 粗排 → 精排 → 重排"}
            </div>
          </div>
          {data ? (
            <div className="flex items-center gap-2 text-[12px] text-mist-300">
              <Clock3 size={14} className="text-gold" />
              {data.total_latency_ms.toFixed(1)} ms
            </div>
          ) : null}
        </div>

        {error ? <div className="glass border-bad/40 p-6 text-[13px] text-bad">{error}</div> : null}

        {!query ? (
          <div className="glass flex flex-col items-center gap-3 p-16 text-center">
            <Search size={26} className="text-gold" />
            <div className="text-[14px] text-mist-100">试试「科幻 太空」或点击左侧热门词</div>
            <div className="text-[12px] text-mist-500">BM25 文本召回 + 向量召回 + ItemCF + 热门，四路融合后进入排序漏斗</div>
          </div>
        ) : null}

        {loading ? (
          <div className="grid gap-4 sm:grid-cols-2 2xl:grid-cols-3">
            {Array.from({ length: 6 }).map((_, index) => (
              <div key={index} className="skeleton h-[380px]" />
            ))}
          </div>
        ) : (
          <div className="grid gap-4 sm:grid-cols-2 2xl:grid-cols-3">
            {(data?.items || []).map((item, index) => (
              <MovieCard key={`${item.movie_id}-${index}`} item={item} index={index} />
            ))}
          </div>
        )}
      </section>

      <aside className="space-y-4">
        <div className="glass p-4">
          <div className="mb-3 flex items-center gap-2 text-[13px] font-semibold text-mist-100">
            <Gauge size={15} className="text-gold" /> 本次请求漏斗
          </div>
          {stages.length ? (
            <>
              <FunnelWaterfall stages={stages} compact />
              <div className="mt-4 space-y-2 border-t border-white/8 pt-3 text-[11px] text-mist-500">
                <div className="flex items-center justify-between">
                  <span className="flex items-center gap-1"><Zap size={12} className="text-gold" /> 端到端</span>
                  <span className="text-mist-300">{data?.total_latency_ms.toFixed(1)} ms</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="flex items-center gap-1"><Sparkles size={12} className="text-aqua" /> 广告位</span>
                  <span className="text-mist-300">{adCount} 条</span>
                </div>
              </div>
            </>
          ) : (
            <div className="py-8 text-center text-[12px] text-mist-500">先发起一次搜索</div>
          )}
        </div>

        {stages.find((stage) => stage.stage === "recall") ? (
          <div className="glass p-4">
            <div className="mb-3 text-[13px] font-semibold text-mist-100">召回各路命中</div>
            <div className="space-y-2">
              {Object.entries(stages[0].sources || {}).map(([source, count]) => (
                <div key={source} className="space-y-1">
                  <div className="flex items-center justify-between text-[11px] text-mist-500">
                    <span>{source}</span>
                    <span className="text-mist-300">{count}</span>
                  </div>
                  <div className="h-1.5 overflow-hidden rounded-full bg-white/8">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-aqua to-gold transition-all duration-700"
                      style={{ width: `${Math.min(100, (count / Math.max(1, Math.max(...Object.values(stages[0].sources)))) * 100)}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        ) : null}
      </aside>
    </div>
  );
}
