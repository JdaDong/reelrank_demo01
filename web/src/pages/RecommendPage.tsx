import { useEffect, useMemo, useState } from "react";
import { Compass, Gauge, History, Sparkles, UserRound } from "lucide-react";
import { api, type RankResponse, type UserBrief } from "../api/client";
import MovieCard from "../components/MovieCard";
import FunnelWaterfall from "../components/FunnelWaterfall";

interface Props {
  userId: number | null;
  users: UserBrief[];
  onUserChange: (userId: number) => void;
}

export default function RecommendPage({ userId, users, onUserChange }: Props) {
  const [data, setData] = useState<RankResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [topk, setTopk] = useState(24);
  const [adsOn, setAdsOn] = useState(true);

  useEffect(() => {
    setLoading(true);
    api
      .recommend({ user_id: userId, topk, ads: adsOn, session_id: "web-recommend" })
      .then(setData)
      .catch((error: Error) => console.error(error.message))
      .finally(() => setLoading(false));
  }, [userId, topk, adsOn]);

  const profile = users.find((user) => user.user_id === userId);
  const organic = (data?.items || []).filter((item) => !item.is_ad);
  const sections = useMemo(
    () => [
      { key: "itemcf", title: "因为你看过", icon: History, items: organic.filter((item) => item.sources.includes("itemcf")).slice(0, 6) },
      { key: "vector", title: "相似口味 · 向量召回", icon: Sparkles, items: organic.filter((item) => item.sources.includes("vector")).slice(0, 6) },
      { key: "explore", title: "探索发现", icon: Compass, items: organic.filter((item) => item.sources.includes("explore") || item.sources.includes("hot")).slice(0, 6) },
    ],
    [organic]
  );

  return (
    <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_330px]">
      <section className="space-y-5">
        <div className="glass flex flex-wrap items-center gap-6 p-5">
          <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br from-gold to-ember text-ink-900 shadow-glow">
            <UserRound size={24} />
          </div>
          <div className="min-w-[220px] flex-1">
            <div className="text-[15px] font-semibold text-mist-100">
              用户 #{userId ?? "-"} 的画像
            </div>
            <div className="mt-1 flex flex-wrap items-center gap-1.5 text-[12px] text-mist-500">
              {(profile?.top_genres || []).map((genre) => (
                <span key={genre} className="chip text-gold">
                  {genre}
                </span>
              ))}
              {!profile?.top_genres?.length ? <span>暂无偏好</span> : null}
            </div>
          </div>
          <div className="flex gap-6 text-[12px] text-mist-500">
            <div>
              <div className="text-[18px] font-bold text-mist-100">{profile ? (profile.click_rate * 100).toFixed(1) : "-"}%</div>
              <div>历史点击率</div>
            </div>
            <div>
              <div className="text-[18px] font-bold text-mist-100">{profile?.history_size ?? "-"}</div>
              <div>正反馈影片</div>
            </div>
          </div>
          <label className="flex items-center gap-2 text-[12px] text-mist-500">
            切换用户
            <select className="field w-[160px] py-1.5" value={userId ?? ""} onChange={(event) => onUserChange(Number(event.target.value))}>
              {users.map((user) => (
                <option key={user.user_id} value={user.user_id}>
                  #{user.user_id}
                </option>
              ))}
            </select>
          </label>
        </div>

        <div className="glass flex items-center justify-between px-5 py-3">
          <div className="text-[13px] text-mist-300">无 query 场景，以用户历史为 trigger 走同一套四层漏斗</div>
          <label className="flex items-center gap-2 text-[12px] text-mist-300">
            结果条数
            <input type="range" min={12} max={40} value={topk} onChange={(event) => setTopk(Number(event.target.value))} className="w-32 accent-gold" />
            <span className="w-6 text-right text-gold">{topk}</span>
          </label>
        </div>

        {loading ? (
          <div className="grid gap-4 sm:grid-cols-2 2xl:grid-cols-3">
            {Array.from({ length: 6 }).map((_, index) => (
              <div key={index} className="skeleton h-[380px]" />
            ))}
          </div>
        ) : (
          sections.map((section) =>
            section.items.length ? (
              <div key={section.key} className="space-y-3">
                <div className="flex items-center gap-2 text-[14px] font-semibold text-mist-100">
                  <section.icon size={16} className="text-gold" />
                  {section.title}
                  <span className="text-[11px] font-normal text-mist-500">{section.items.length} 部</span>
                </div>
                <div className="grid gap-4 sm:grid-cols-2 2xl:grid-cols-3">
                  {section.items.map((item, index) => (
                    <MovieCard key={`${section.key}-${item.movie_id}-${index}`} item={item} index={index} />
                  ))}
                </div>
              </div>
            ) : null
          )
        )}
      </section>

      <aside className="space-y-4">
        <div className="glass p-4">
          <div className="mb-3 flex items-center gap-2 text-[13px] font-semibold text-mist-100">
            <Gauge size={15} className="text-gold" /> 本次请求漏斗
          </div>
          {data?.trace ? (
            <FunnelWaterfall stages={data.trace.stages} compact />
          ) : (
            <div className="py-8 text-center text-[12px] text-mist-500">加载中…</div>
          )}
        </div>
        <div className="glass p-4">
          <div className="mb-3 text-[13px] font-semibold text-mist-100">广告与自然结果</div>
          <div className="space-y-2 text-[12px] text-mist-500">
            <div className="flex items-center justify-between">
              <span>自然结果</span>
              <span className="text-mist-100">{organic.length}</span>
            </div>
            <div className="flex items-center justify-between">
              <span>广告条数</span>
              <span className="text-gold">{(data?.items || []).filter((item) => item.is_ad).length}</span>
            </div>
            <div className="flex items-center justify-between">
              <span>端到端耗时</span>
              <span className="text-mist-100">{data ? `${data.total_latency_ms.toFixed(1)} ms` : "-"}</span>
            </div>
          </div>
          <label className="mt-3 flex items-center justify-between border-t border-white/8 pt-3 text-[12px] text-mist-300">
            投放广告
            <input type="checkbox" checked={adsOn} onChange={(event) => setAdsOn(event.target.checked)} className="h-4 w-4 accent-gold" />
          </label>
        </div>
      </aside>
    </div>
  );
}
