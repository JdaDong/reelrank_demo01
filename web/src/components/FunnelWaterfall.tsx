import { Filter, Layers, ListOrdered, Megaphone, Target } from "lucide-react";
import type { StageTrace } from "../api/client";

const META: Record<string, { label: string; color: string; icon: typeof Filter }> = {
  recall: { label: "召回", color: "from-aqua to-[#7CC4FF]", icon: Filter },
  coarse: { label: "粗排", color: "from-[#7CC4FF] to-gold", icon: Layers },
  fine: { label: "精排", color: "from-gold to-ember", icon: Target },
  rerank: { label: "重排", color: "from-ember to-[#EF4444]", icon: ListOrdered },
  ads: { label: "广告竞价", color: "from-gold/70 to-gold", icon: Megaphone },
};

interface Props {
  stages: StageTrace[];
  activeStage?: string;
  onSelect?: (stage: string) => void;
  compact?: boolean;
}

export default function FunnelWaterfall({ stages, activeStage, onSelect, compact = false }: Props) {
  if (!stages.length) return null;
  const max = Math.max(...stages.map((stage) => Math.max(stage.input_count, stage.output_count)), 1);

  return (
    <div className="space-y-3">
      {stages.map((stage, index) => {
        const meta = META[stage.stage] || META.recall;
        const Icon = meta.icon;
        const width = Math.max(6, (stage.output_count / max) * 100);
        const active = activeStage === stage.stage;
        return (
          <button
            key={stage.stage}
            type="button"
            onClick={() => onSelect?.(stage.stage)}
            className={`w-full rounded-xl border px-3 py-2.5 text-left transition-all duration-200 ${
              active ? "border-gold/50 bg-gold/10" : "border-white/8 bg-white/[0.02] hover:border-white/20 hover:bg-white/[0.05]"
            }`}
          >
            <div className="flex items-center justify-between text-[12px]">
              <span className="flex items-center gap-2 font-semibold text-mist-100">
                <Icon size={14} className="text-gold" />
                {index + 1}. {meta.label}
              </span>
              <span className="text-mist-300">
                {stage.input_count} → <span className="font-semibold text-gold">{stage.output_count}</span>
              </span>
            </div>
            <div className="mt-2 h-2.5 overflow-hidden rounded-full bg-white/8">
              <div
                className={`h-full rounded-full bg-gradient-to-r ${meta.color} transition-all duration-700`}
                style={{ width: `${width}%` }}
              />
            </div>
            {!compact ? (
              <div className="mt-2 flex items-center justify-between text-[11px] text-mist-500">
                <span className="line-clamp-1">{stage.cut_reason}</span>
                <span className="shrink-0 pl-2 text-mist-300">{stage.latency_ms.toFixed(1)}ms</span>
              </div>
            ) : null}
          </button>
        );
      })}
    </div>
  );
}
