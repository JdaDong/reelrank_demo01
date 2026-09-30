const BASE = "";

async function getJSON<T>(path: string, params?: Record<string, string | number | boolean | undefined | null>): Promise<T> {
  const url = new URL(`${BASE}${path}`, window.location.origin);
  Object.entries(params || {}).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") {
      url.searchParams.set(key, String(value));
    }
  });
  const response = await fetch(url.toString(), { headers: { accept: "application/json" } });
  if (!response.ok) {
    throw new Error(`${path} 请求失败：${response.status}`);
  }
  return (await response.json()) as T;
}

async function postJSON<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json", accept: "application/json" },
    body: JSON.stringify(body || {}),
  });
  if (!response.ok) {
    throw new Error(`${path} 请求失败：${response.status}`);
  }
  return (await response.json()) as T;
}

export interface ResultItem {
  movie_id: number;
  title: string;
  year: number;
  genres: string[];
  vote_average: number;
  popularity: number;
  poster_path: string;
  overview: string;
  score: number;
  pctr: number;
  sources: string[];
  is_ad: boolean;
  ad: AdItem | null;
  rank_before: number | null;
}

export interface CandidateItem {
  movie_id: number;
  title: string;
  scores: Record<string, number>;
  sources: string[];
  rank: number | null;
  rank_before: number | null;
  cut_reason: string | null;
}

export interface StageTrace {
  stage: "recall" | "coarse" | "fine" | "rerank" | "ads";
  input_count: number;
  output_count: number;
  latency_ms: number;
  cut_reason: string;
  items: CandidateItem[];
  sources: Record<string, number>;
}

export interface AdItem {
  ad_id: number;
  advertiser_id: number;
  advertiser_name: string;
  movie_id: number;
  title: string;
  slot: number | null;
  pctr: number;
  bid: number;
  quality_factor: number;
  ecpm: number;
  price: number;
  blocked_reason: string | null;
}

export interface FunnelTrace {
  request_id: string;
  scene: string;
  query: string | null;
  user_id: number | null;
  stages: StageTrace[];
  ads: AdItem[];
  total_latency_ms: number;
  config: Record<string, number>;
}

export interface RankResponse {
  request_id: string;
  scene: string;
  query: string | null;
  user_id: number | null;
  items: ResultItem[];
  trace: FunnelTrace | null;
  total_latency_ms: number;
}

export interface UserBrief {
  user_id: number;
  top_genres: string[];
  click_rate: number;
  history_size: number;
}

export interface AdvertiserRow {
  advertiser_id: number;
  advertiser_name: string;
  bid: number;
  daily_budget: number;
  spent: number;
  freq_cap: number;
  creatives: number;
  targeting: string[];
  budget_usage: number;
}

export interface StatsResponse {
  funnel: {
    recall: Record<string, number>;
    coarse: Record<string, number>;
    fine: { topk: number; weights: Record<string, number> };
    rerank: Record<string, number>;
  };
  ads: { enabled: boolean; slots: number[]; gsp: boolean };
  data_source: string | null;
  behavior: number | null;
  train: { coarse_auc: number; fine_valid_auc: number | null } | null;
  top_features: [string, number][];
}

export interface HealthResponse {
  status: string;
  movies: number;
  users: number;
  ads: number;
  coarse_auc: number;
  fine_auc: number | null;
  vector_explained_variance: number;
}

export interface MovieDetail {
  movie_id: number;
  title: string;
  year: number;
  genres: string[];
  vote_average: number;
  popularity: number;
  poster_path: string;
  overview: string;
  features: Record<string, number>;
  similar: { movie_id: number; title: string; score: number }[];
}

export const api = {
  health: () => getJSON<HealthResponse>("/health"),
  stats: () => getJSON<StatsResponse>("/api/stats"),
  users: (limit = 40) => getJSON<{ users: UserBrief[] }>("/api/users", { limit }),
  movie: (movieId: number) => getJSON<MovieDetail>(`/api/movies/${movieId}`),
  search: (params: { query: string; user_id?: number | null; topk?: number | null; ads?: boolean; session_id?: string }) =>
    getJSON<RankResponse>("/api/search", { ...params, trace: true }),
  recommend: (params: { user_id?: number | null; topk?: number | null; ads?: boolean; session_id?: string }) =>
    getJSON<RankResponse>("/api/recommend", { ...params, trace: true }),
  funnelDebug: (payload: Record<string, unknown>) => postJSON<RankResponse>("/api/funnel/debug", payload),
  adsBoard: () =>
    getJSON<{ date: string; slots: number[]; advertisers: AdvertiserRow[]; total_spent: number; total_budget: number }>("/api/ads/board"),
  adsReset: () => postJSON<{ status: string; ads: number }>("/api/ads/reset", {}),
};

export function posterUrl(path: string): string {
  return path ? `https://image.tmdb.org/t/p/w342${path}` : "";
}
