import { useEffect, useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Gauge, Play, RotateCcw, Timer } from "lucide-react";
import { api, type RankResponse, type StageTrace, type UserBrief } from "../api/client";
import FunnelWaterfall from "../components/FunnelWaterfall";
import StageTable from "../components/StageTable";

interface Props {
  userId: number | null;
  users: UserBrief[];
  onUserChange: (userId: number) => void;
}

const STAGE_SCORES: Record<string, string[]> = {
  recall: ["bm25_score", "vector_score", "itemcf_score", "hot_score"],
  coarse: ["bm25_score", "vector_score", "itemcf_score", "lr_score", "coarse_score"],
  fine: ["coarse_score", "pctr", "fine_score"],
  rerank: ["pctr", "fine_score"],
};

export default function FunnelPage({ userId, users, onUserChange }: Props) {
  const [scene, setScene] = useState<"search" | "recommend">("search");
  const [query, setQuery] = useState("科幻 太空");
  const [recallLimit, setRecallLimit] = useState(600);
  const [coarseTopk, setCoarseTopk] = useState(200);
  const [fineTopk, setFineTopk] = useState(30);
  const [topk, setTopk] = useState(12);
  const [adsOn, setAdsOn] = useState(true);
  const [data, setData] = useState<RankResponse | null>(null);
  const [activeStage, setActiveStage] = useState<string>("recall");
  const [loading, setLoading] = useState(false);
  const [features, setFeatures] = useState<[string, number][]>([]);

  useEffect(() => {
    api.stats().then((stats) => setFeatures(stats.top_features || [])).catch((error: Error) => console.error(error.message));
  }, []);

  const run = () => {
    setLoading(true);
    api
      .funnelDebug({
        scene,
        query: scene === "search" ? query : null,
        user_id: userId,
        recall_limit: recallLimit,
        coarse_topk: coarseTopk,
        fine_topk: fineTopk,
        topk,
        ads_enabled: adsOn,
        session_id: "web-funnel",
      })
      .then(setData)
      .catch((error: Error) => console.error(error.message))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    run();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const stages: StageTrace[] = data?.trace?.stages ?? [];
  const current = useMemo(() => stages.find((stage) => stage.stage === activeStage), [stages, activeStage]);
  const latencyData = stages.map((stage) => ({ stage: stage.stage, ms: Number(stage.latency_ms.toFixed(2)) }));

  return (
    <div className="space-y-5">
      <div className="glass space-y-4 p-5">
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-2 text-[15px] font-semibold text-mist-100">
            <Gauge size={17} className="text-gold" /> 漏斗调试面板
          </div>
          <div className="flex rounded-xl border border-white/10 bg-white/5 p-1">
            {(["search", "recommend"] as const).map((item) => (
              <button
                key={item}
                type="button"
                onClick={() => setScene(item)}
                className={`rounded-lg px-3 py-1.5 text-[12px] font-medium transition-colors ${
                  scene === item ? "bg-gold/20 text-gold" : "text-mist-300 hover:text-mist-100"
                }`}
              >
                {item === "search" ? "搜索场景" : "推荐场景"}
              </button>
            ))}
          </div>
          {scene === "search" ? (
            <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="输入 query" className="field w-[240px]" />
          ) : null}
          <label className="flex items-center gap-2 text-[12px] text-mist-500">
            用户
            <select className="field w-[130px] py-1.5" value={userId ?? ""} onChange={(event) => onUserChange(Number(event.target.value))}>
              {users.map((user) => (
                <option key={user.user_id} value={user.user_id}>
                  #{user.user_id}
                </option>
              ))}
            </select>
          </label>
          <label className="flex items-center gap-2 text-[12px] text-mist-300">
            广告
            <input type="checkbox" checked={adsOn} onChange={(event) => setAdsOn(event.target.checked)} className="h-4 w-4 accent-gold" />
          </label>
          <button type="button" onClick={run} disabled={loading} className="btn-primary ml-auto">
            <Play size={14} /> {loading ? "执行中…" : "执行一次请求"}
          </button>
        </div>

        <div className="grid gap-4 md:grid-cols-4">
          {[
            { label: "召回上限", value: recallLimit, min: 100, max: 1000, step: 50, setter: setRecallLimit },
            { label: "粗排截断", value: coarseTopk, min: 50, max: 500, step: 10, setter: setCoarseTopk },
            { label: "精排截断", value: fineTopk, min: 10, max: 100, step: 5, setter: setFineTopk },
            { label: "重排输出", value: topk, min: 4, max: 30, step: 1, setter: setTopk },
          ].map((control) => (
            <label key={control.label} className="space-y-2 rounded-xl border border-white/8 bg-white/[0.02] p-3">
              <span className="flex items-center justify-between text-[12px] text-mist-500">
                {control.label}
                <span className="font-semibold text-gold">{control.value}</span>
              </span>
              <input
                type="range"
                min={control.min}
                max={control.max}
                step={control.step}
                value={control.value}
                onChange={(event) => control.setter(Number(event.target.value))}
                className="w-full accent-gold"
              />
            </label>
          ))}
        </div>
      </div>

      <div className="grid gap-5 xl:grid-cols-[360px_minmax(0,1fr)]">
        <aside className="space-y-4">
          <div className="glass p-4">
            <div className="mb-3 text-[13px] font-semibold text-mist-100">逐层收敛</div>
            {stages.length ? (
              <FunnelWaterfall stages={stages} activeStage={activeStage} onSelect={setActiveStage} />
            ) : (
              <div className="py-10 text-center text-[12px] text-mist-500">点击「执行一次请求」开始</div>
            )}
          </div>

          <div className="glass p-4">
            <div className="mb-3 flex items-center gap-2 text-[13px] font-semibold text-mist-100">
              <Timer size={15} className="text-gold" /> 各层耗时
            </div>
            <div className="h-[190px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={latencyData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                  <XAxis dataKey="stage" stroke="#6B7488" fontSize={11} />
                  <YAxis stroke="#6B7488" fontSize={11} unit="ms" />
                  <Tooltip
                    contentStyle={{ background: "#12161F", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 12, fontSize: 12 }}
                    formatter={(value: number) => [`${value} ms`, "耗时"]}
                  />
                  <Bar dataKey="ms" fill="#FFB020" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {features.length ? (
            <div className="glass p-4">
              <div className="mb-3 text-[13px] font-semibold text-mist-100">精排特征重要性（Top）</div>
              <div className="space-y-2">
                {features.slice(0, 7).map(([name, value]) => (
                  <div key={name} className="space-y-1">
                    <div className="flex items-center justify-between text-[11px] text-mist-500">
                      <span>{name}</span>
                      <span className="text-mist-300">{value.toFixed(4)}</span>
                    </div>
                    <div className="h-1.5 overflow-hidden rounded-full bg-white/8">
                      <div
                        className="h-full rounded-full bg-gradient-to-r from-aqua to-gold"
                        style={{ width: `${Math.min(100, (value / Math.max(0.0001, features[0][1])) * 100)}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : null}
        </aside>

        <section className="space-y-4">
          <div className="glass p-5">
            <div className="mb-3 flex items-center justify-between">
              <div className="text-[14px] font-semibold text-mist-100">
                {activeStage === "ads" ? "广告竞价明细" : `${activeStage} 层候选明细`}
              </div>
              <div className="flex items-center gap-3 text-[11px] text-mist-500">
                <span>request {data?.request_id || "-"}</span>
                <span className="flex items-center gap-1">
                  <RotateCcw size={11} /> 总耗时 {data ? `${data.total_latency_ms.toFixed(1)} ms` : "-"}
                </span>
              </div>
            </div>

            {activeStage === "ads" ? (
              <div className="overflow-hidden rounded-xl border border-white/8">
                <table className="w-full min-w-[860px] text-left text-[12px]">
                  <thead className="bg-ink-800/95 text-mist-500">
                    <tr>
                      <th className="px-4 py-2 font-medium">广告主</th>
                      <th className="px-4 py-2 font-medium">创意</th>
                      <th className="px-4 py-2 font-medium">pCTR</th>
                      <th className="px-4 py-2 font-medium">出价</th>
                      <th className="px-4 py-2 font-medium">质量因子</th>
                      <th className="px-4 py-2 font-medium">eCPM</th>
                      <th className="px-4 py-2 font-medium">二价扣费</th>
                      <th className="px-4 py-2 font-medium">广告位</th>
                      <th className="px-4 py-2 font-medium">状态</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(data?.trace?.ads || []).map((ad) => (
                      <tr key={ad.ad_id} className="border-t border-white/6 hover:bg-white/[0.04]">
                        <td className="px-4 py-2 text-mist-100">{ad.advertiser_name}</td>
                        <td className="px-4 py-2 text-mist-300">{ad.title}</td>
                        <td className="px-4 py-2 text-mist-300">{(ad.pctr * 100).toFixed(2)}%</td>
                        <td className="px-4 py-2 text-mist-300">¥{ad.bid}</td>
                        <td className="px-4 py-2 text-mist-300">{ad.quality_factor}</td>
                        <td className="px-4 py-2 font-semibold text-gold">{ad.ecpm.toFixed(1)}</td>
                        <td className="px-4 py-2 text-mist-300">{ad.blocked_reason ? "-" : `¥${ad.price}`}</td>
                        <td className="px-4 py-2 text-mist-300">{ad.slot ?? "-"}</td>
                        <td className="px-4 py-2">
                          {ad.blocked_reason ? (
                            <span className="text-bad">{ad.blocked_reason}</span>
                          ) : (
                            <span className="text-ok">已中标</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <StageTable
                items={current?.items || []}
                scoreKeys={STAGE_SCORES[activeStage] || []}
                title={`${current?.output_count ?? 0} 条候选 · ${current?.cut_reason ?? ""}`}
              />
            )}
          </div>

          <div className="glass p-5">
            <div className="mb-3 text-[14px] font-semibold text-mist-100">最终混排结果</div>
            <div className="grid gap-3 md:grid-cols-2">
              {(data?.items || []).map((item, index) => (
                <div
                  key={`${item.movie_id}-${index}`}
                  className={`flex items-center gap-3 rounded-xl border px-3 py-2.5 ${
                    item.is_ad ? "border-gold/45 bg-gold/[0.08]" : "border-white/8 bg-white/[0.02]"
                  }`}
                >
                  <span className="w-8 text-[12px] font-semibold text-gold">#{index + 1}</span>
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-[13px] text-mist-100">{item.title}</div>
                    <div className="text-[11px] text-mist-500">
                      {item.is_ad ? `${item.ad?.advertiser_name} · eCPM ${item.ad?.ecpm.toFixed(1)}` : (item.sources || []).join("/")}
                    </div>
                  </div>
                  {item.rank_before !== null && item.rank_before !== index ? (
                    <span className="text-[11px] text-mist-500">重排前 #{item.rank_before + 1}</span>
                  ) : null}
                </div>
              ))}
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
