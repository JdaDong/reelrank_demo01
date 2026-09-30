import type { LucideIcon } from "lucide-react";

interface Props {
  label: string;
  value: string | number;
  hint?: string;
  icon: LucideIcon;
  tone?: "gold" | "aqua" | "ok" | "warn" | "bad";
}

const TONE: Record<string, string> = {
  gold: "text-gold",
  aqua: "text-aqua",
  ok: "text-ok",
  warn: "text-warn",
  bad: "text-bad",
};

export default function MetricCard({ label, value, hint, icon: Icon, tone = "gold" }: Props) {
  return (
    <div className="glass glass-hover p-4">
      <div className="flex items-center justify-between">
        <span className="text-[12px] text-mist-500">{label}</span>
        <Icon size={16} className={TONE[tone]} />
      </div>
      <div className="mt-2 text-[26px] font-bold leading-none text-mist-100">{value}</div>
      {hint ? <div className="mt-1.5 text-[11px] text-mist-500">{hint}</div> : null}
    </div>
  );
}
