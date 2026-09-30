import { ArrowDown, ArrowUp, Minus } from "lucide-react";
import type { CandidateItem } from "../api/client";

interface Props {
  items: CandidateItem[];
  scoreKeys: string[];
  title?: string;
}

const KEY_LABEL: Record<string, string> = {
  bm25_score: "BM25",
  vector_score: "向量",
  itemcf_score: "ItemCF",
  hot_score: "热门",
  lr_score: "LR",
  coarse_score: "粗排分",
  pctr: "pCTR",
  fine_score: "精排分",
};

export default function StageTable({ items, scoreKeys, title }: Props) {
  if (!items.length) {
    return <div className="rounded-xl border border-white/8 bg-white/[0.02] p-6 text-center text-[12px] text-mist-500">该层暂无候选明细</div>;
  }

  return (
    <div className="overflow-hidden rounded-xl border border-white/8">
      {title ? (
        <div className="border-b border-white/8 bg-white/[0.03] px-4 py-2.5 text-[12px] font-semibold text-mist-100">{title}</div>
      ) : null}
      <div className="max-h-[420px] overflow-auto">
        <table className="w-full min-w-[760px] text-left text-[12px]">
          <thead className="sticky top-0 bg-ink-800/95 text-mist-500 backdrop-blur">
            <tr>
              <th className="px-4 py-2 font-medium">#</th>
              <th className="px-4 py-2 font-medium">影片</th>
              <th className="px-4 py-2 font-medium">来源</th>
              {scoreKeys.map((key) => (
                <th key={key} className="px-4 py-2 font-medium">
                  {KEY_LABEL[key] || key}
                </th>
              ))}
              <th className="px-4 py-2 font-medium">位次变化</th>
            </tr>
          </thead>
          <tbody>
            {items.map((item, index) => {
              const before = item.rank_before ?? null;
              const delta = before === null ? null : before - index;
              return (
                <tr key={`${item.movie_id}-${index}`} className="border-t border-white/6 hover:bg-white/[0.04]">
                  <td className="px-4 py-2 text-mist-500">{index + 1}</td>
                  <td className="px-4 py-2 text-mist-100">{item.title || item.movie_id}</td>
                  <td className="px-4 py-2 text-mist-500">{(item.sources || []).join("/") || "-"}</td>
                  {scoreKeys.map((key) => (
                    <td key={key} className="px-4 py-2 text-mist-300">
                      {(item.scores?.[key] ?? 0).toFixed(3)}
                    </td>
                  ))}
                  <td className="px-4 py-2">
                    {delta === null ? (
                      <span className="text-mist-500">-</span>
                    ) : delta > 0 ? (
                      <span className="flex items-center gap-1 text-ok">
                        <ArrowUp size={12} /> {delta}
                      </span>
                    ) : delta < 0 ? (
                      <span className="flex items-center gap-1 text-bad">
                        <ArrowDown size={12} /> {Math.abs(delta)}
                      </span>
                    ) : (
                      <span className="flex items-center gap-1 text-mist-500">
                        <Minus size={12} /> 0
                      </span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
