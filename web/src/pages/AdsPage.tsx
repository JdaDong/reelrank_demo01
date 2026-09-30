import { useCallback, useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { BadgeCheck, Coins, Megaphone, RefreshCw, Target, Wallet } from "lucide-react";
import { api, type AdvertiserRow, type RankResponse } from "../api/client";
import MetricCard from "../components/MetricCard";

interface Props {
  userId: number | null;
}

interface Board {
  date: string;
  slots: number[];
  advertisers: AdvertiserRow[];
  total_spent: number;
  total_budget: number;
}

export default function AdsPage({ userId }: Props) {
  const [board, setBoard] = useState<Board | null>(null);
  const [latest, setLatest] = useState<RankResponse | null>(null);
  const [loading, setLoading] = useState(false);

  const refresh = useCallback(() => {
    api.adsBoard().then(setBoard).catch((error: Error) => console.error(error.message));
  }, []);

  const runOnce = useCallback(() => {
    setLoading(true);
    api
      .funnelDebug({ scene: "recommend", user_id: userId, topk: 20, ads_enabled: true, session_id: "web-ads" })
      .then((response) => {
        setLatest(response);
        refresh();
      })
      .catch((error: Error) => console.error(error.message))
      .finally(() => setLoading(false));
  }, [userId, refresh]);

  useEffect(() => {
    refresh();
    runOnce();
  }, [refresh, runOnce]);

  const winners = (latest?.trace?.ads || []).filter((ad) => !ad.blocked_reason);
  const blocked = (latest?.trace?.ads || []).filter((ad) => ad.blocked_reason);
  const avgEcpm = winners.length ? winners.reduce((sum, ad) => sum + ad.ecpm, 0) / winners.length : 0;
  const avgPrice = winners.length ? winners.reduce((sum, ad) => sum + ad.price, 0) / winners.length : 0;
  const usage = board && board.total_budget ? (board.total_spent / board.total_budget) * 100 : 0;

  return (
    <div className="space-y-5">
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <MetricCard label="本次曝光广告" value={winners.length} hint={`广告位 ${(board?.slots || []).join(" / ")}`} icon={BadgeCheck} tone="gold" />
        <MetricCard label="平均 eCPM" value={avgEcpm.toFixed(1)} hint="元 / 千次曝光" icon={Target} tone="aqua" />
        <MetricCard label="平均二价扣费" value={`¥${avgPrice.toFixed(2)}`} hint="GSP 二价结算" icon={Coins} tone="ok" />
        <MetricCard
          label="日预算消耗"
          value={`${usage.toFixed(1)}%`}
          hint={board ? `¥${board.total_spent} / ¥${board.total_budget}` : "-"}
          icon={Wallet}
          tone={usage > 80 ? "bad" : "warn"}
        />
      </div>

      <div className="glass flex flex-wrap items-center justify-between gap-3 px-5 py-4">
        <div className="flex items-center gap-2 text-[14px] font-semibold text-mist-100">
          <Megaphone size={16} className="text-gold" /> 广告投放看板
          <span className="text-[11px] font-normal text-mist-500">eCPM = pCTR × bid × 质量因子 × 1000，GSP 二价扣费</span>
        </div>
        <div className="flex gap-2">
          <button type="button" onClick={runOnce} disabled={loading} className="btn-primary">
            <RefreshCw size={14} /> {loading ? "请求中…" : "再跑一次请求"}
          </button>
          <button
            type="button"
            onClick={() => {
              api.adsReset().then(refresh).catch((error: Error) => console.error(error.message));
            }}
            className="btn-ghost"
          >
            重置预算
          </button>
        </div>
      </div>

      <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <div className="glass p-5">
          <div className="mb-3 text-[13px] font-semibold text-mist-100">广告主列表（{board?.date ?? "-"}）</div>
          <div className="max-h-[420px] overflow-auto">
            <table className="w-full min-w-[720px] text-left text-[12px]">
              <thead className="sticky top-0 bg-ink-800/95 text-mist-500">
                <tr>
                  <th className="px-3 py-2 font-medium">广告主</th>
                  <th className="px-3 py-2 font-medium">出价</th>
                  <th className="px-3 py-2 font-medium">日预算</th>
                  <th className="px-3 py-2 font-medium">已消耗</th>
                  <th className="px-3 py-2 font-medium">消耗率</th>
                  <th className="px-3 py-2 font-medium">创意 / 频次</th>
                  <th className="px-3 py-2 font-medium">定向</th>
                </tr>
              </thead>
              <tbody>
                {(board?.advertisers || []).map((row) => (
                  <tr key={row.advertiser_id} className="border-t border-white/6 hover:bg-white/[0.04]">
                    <td className="px-3 py-2 text-mist-100">{row.advertiser_name}</td>
                    <td className="px-3 py-2 text-mist-300">¥{row.bid}</td>
                    <td className="px-3 py-2 text-mist-300">¥{row.daily_budget}</td>
                    <td className="px-3 py-2 text-gold">¥{row.spent}</td>
                    <td className="px-3 py-2">
                      <div className="flex items-center gap-2">
                        <div className="h-1.5 w-16 overflow-hidden rounded-full bg-white/8">
                          <div
                            className={`h-full rounded-full ${row.budget_usage > 0.8 ? "bg-bad" : "bg-gradient-to-r from-gold to-ember"}`}
                            style={{ width: `${Math.min(100, row.budget_usage * 100)}%` }}
                          />
                        </div>
                        <span className="text-mist-500">{(row.budget_usage * 100).toFixed(1)}%</span>
                      </div>
                    </td>
                    <td className="px-3 py-2 text-mist-500">
                      {row.creatives} / {row.freq_cap}
                    </td>
                    <td className="px-3 py-2 text-mist-500">{(row.targeting || []).join("、") || "不限"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="space-y-5">
          <div className="glass p-5">
            <div className="mb-3 text-[13px] font-semibold text-mist-100">本次竞价结果</div>
            {winners.length ? (
              <div className="space-y-2">
                {winners.map((ad) => (
                  <div key={ad.ad_id} className="rounded-xl border border-gold/35 bg-gold/[0.06] p-3">
                    <div className="flex items-center justify-between text-[13px] text-mist-100">
                      <span className="font-semibold">{ad.advertiser_name}</span>
                      <span className="text-gold">eCPM {ad.ecpm.toFixed(1)}</span>
                    </div>
                    <div className="mt-1 text-[11px] text-mist-500">
                      {ad.title} · 广告位 #{ad.slot} · pCTR {(ad.pctr * 100).toFixed(2)}% · 出价 ¥{ad.bid} → 二价 ¥{ad.price}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="py-6 text-center text-[12px] text-mist-500">本次未产生中标广告</div>
            )}
            {blocked.length ? (
              <div className="mt-3 space-y-1 border-t border-white/8 pt-3 text-[11px] text-mist-500">
                {blocked.slice(0, 6).map((ad) => (
                  <div key={ad.ad_id} className="flex items-center justify-between">
                    <span>{ad.advertiser_name}</span>
                    <span className="text-bad">{ad.blocked_reason}</span>
                  </div>
                ))}
              </div>
            ) : null}
          </div>

          <div className="glass p-5">
            <div className="mb-3 text-[13px] font-semibold text-mist-100">各广告主预算消耗</div>
            <div className="h-[240px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={(board?.advertisers || []).map((row) => ({ name: row.advertiser_name, spent: row.spent, budget: row.daily_budget }))}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                  <XAxis dataKey="name" stroke="#6B7488" fontSize={10} interval={0} angle={-18} textAnchor="end" height={52} />
                  <YAxis stroke="#6B7488" fontSize={11} />
                  <Tooltip
                    contentStyle={{ background: "#12161F", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 12, fontSize: 12 }}
                    formatter={(value: number, name: string) => [`¥${value}`, name === "spent" ? "已消耗" : "日预算"]}
                  />
                  <Bar dataKey="budget" fill="rgba(255,255,255,0.14)" radius={[6, 6, 0, 0]} />
                  <Bar dataKey="spent" fill="#FFB020" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
