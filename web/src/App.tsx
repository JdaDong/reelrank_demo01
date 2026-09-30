import { useEffect, useState } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { Activity } from "lucide-react";
import NavBar from "./components/NavBar";
import SearchPage from "./pages/SearchPage";
import RecommendPage from "./pages/RecommendPage";
import FunnelPage from "./pages/FunnelPage";
import AdsPage from "./pages/AdsPage";
import { api, type HealthResponse, type UserBrief } from "./api/client";

export default function App() {
  const [userId, setUserId] = useState<number | null>(null);
  const [users, setUsers] = useState<UserBrief[]>([]);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [offline, setOffline] = useState(false);

  useEffect(() => {
    api
      .health()
      .then(setHealth)
      .catch(() => setOffline(true));
    api
      .users(40)
      .then((response) => {
        setUsers(response.users);
        if (response.users.length) {
          setUserId(response.users[0].user_id);
        }
      })
      .catch(() => setOffline(true));
  }, []);

  return (
    <div className="min-h-screen">
      <NavBar userId={userId} users={users} health={health} onUserChange={setUserId} />

      <main className="mx-auto max-w-[1600px] px-6 pb-20 pt-[92px]">
        {offline ? (
          <div className="glass flex items-center gap-3 border-bad/40 p-5 text-[13px] text-bad">
            <Activity size={16} /> 后端服务未就绪：请先启动 `python -m reelrank.serving.api`（默认 8000 端口）
          </div>
        ) : null}

        <Routes>
          <Route path="/" element={<Navigate to="/search" replace />} />
          <Route path="/search" element={<SearchPage userId={userId} />} />
          <Route path="/recommend" element={<RecommendPage userId={userId} users={users} onUserChange={setUserId} />} />
          <Route path="/funnel" element={<FunnelPage userId={userId} users={users} onUserChange={setUserId} />} />
          <Route path="/ads" element={<AdsPage userId={userId} />} />
          <Route path="*" element={<Navigate to="/search" replace />} />
        </Routes>
      </main>

      <footer className="border-t border-white/8 px-6 py-4 text-center text-[11px] text-mist-500">
        ReelRank · 数据源 TMDB · 四层漏斗（召回 → 粗排 → 精排 → 重排）+ 广告竞价混排
      </footer>
    </div>
  );
}
