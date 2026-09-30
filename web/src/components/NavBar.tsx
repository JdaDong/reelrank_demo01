import { useEffect, useState } from "react";
import { Link, NavLink, useNavigate, useLocation } from "react-router-dom";
import { Clapperboard, Search, Sparkles, Gauge, Megaphone } from "lucide-react";
import type { HealthResponse, UserBrief } from "../api/client";

interface Props {
  userId: number | null;
  users: UserBrief[];
  health: HealthResponse | null;
  onUserChange: (userId: number) => void;
}

const LINKS = [
  { to: "/search", label: "搜索", icon: Search },
  { to: "/recommend", label: "推荐", icon: Sparkles },
  { to: "/funnel", label: "漏斗调试", icon: Gauge },
  { to: "/ads", label: "广告看板", icon: Megaphone },
];

export default function NavBar({ userId, users, health, onUserChange }: Props) {
  const navigate = useNavigate();
  const location = useLocation();
  const [keyword, setKeyword] = useState("");

  useEffect(() => {
    const params = new URLSearchParams(location.search);
    setKeyword(params.get("q") || "");
  }, [location.search]);

  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    if (keyword.trim()) {
      navigate(`/search?q=${encodeURIComponent(keyword.trim())}`);
    }
  };

  return (
    <header className="fixed inset-x-0 top-0 z-40 border-b border-white/10 bg-ink-900/85 backdrop-blur-xl">
      <div className="mx-auto flex h-[72px] max-w-[1600px] items-center gap-6 px-6">
        <Link to="/search" className="flex items-center gap-2">
          <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-gold to-ember text-ink-900 shadow-glow">
            <Clapperboard size={18} />
          </span>
          <div className="leading-tight">
            <div className="text-[15px] font-bold tracking-wide text-mist-100">ReelRank</div>
            <div className="text-[10px] text-mist-500">召回 · 粗排 · 精排 · 重排</div>
          </div>
        </Link>

        <form onSubmit={submit} className="relative max-w-[340px] flex-1">
          <input
            value={keyword}
            onChange={(event) => setKeyword(event.target.value)}
            placeholder="搜索影片、类型、演员…  如：科幻 太空"
            className="field pl-9"
          />
          <Search size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-mist-500" />
        </form>

        <nav className="flex items-center gap-1">
          {LINKS.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                `flex items-center gap-1.5 rounded-xl px-3 py-2 text-[13px] font-medium transition-colors ${
                  isActive ? "bg-gold/15 text-gold" : "text-mist-300 hover:bg-white/5 hover:text-mist-100"
                }`
              }
            >
              <Icon size={15} />
              {label}
            </NavLink>
          ))}
        </nav>

        <div className="ml-auto flex items-center gap-3">
          <label className="flex items-center gap-2 text-[12px] text-mist-500">
            用户
            <select
              className="field w-[150px] py-1.5"
              value={userId ?? ""}
              onChange={(event) => onUserChange(Number(event.target.value))}
            >
              {users.map((user) => (
                <option key={user.user_id} value={user.user_id}>
                  #{user.user_id} · {(user.top_genres || []).slice(0, 2).join("/") || "综合"}
                </option>
              ))}
            </select>
          </label>
          {health ? (
            <div className="hidden items-center gap-2 text-[11px] text-mist-500 xl:flex">
              <span className="chip">{health.movies} 部影片</span>
              <span className="chip">{health.ads} 条广告</span>
              <span className="chip text-ok">精排 AUC {health.fine_auc?.toFixed(3) ?? "-"}</span>
            </div>
          ) : null}
        </div>
      </div>
    </header>
  );
}
