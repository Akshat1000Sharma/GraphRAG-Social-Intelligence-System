const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`API ${res.status}: ${err}`);
  }
  return res.json();
}

// ─── Types ────────────────────────────────────────────────────────────────────

export interface HealthResponse {
  status: string;
  neo4j_connected: boolean;
  vector_backend?: string;
  vector_indexes?: { name: string; state: string }[];
  gnn_loaded: boolean;
  gnn_datasets: string[];
  pipeline_ready: boolean;
  version: string;
  dataset_counts?: Record<string, Record<string, number>>;
  ingest_results?: Record<string, string>;
}

export interface PipelineResult {
  intent: string;
  query: string;
  results: Record<string, unknown>[];
  gnn_predictions: Record<string, unknown>[];
  insight: string;
  graph_context: string;
  retrieval_mode: string;
  sources: string[];
  validation?: {
    is_valid: boolean;
    confidence: number;
    warnings: string[];
    issues: string[];
  };
  pipeline_timing_ms?: {
    analyzer: number;
    router: number;
    retrieval: number;
    gnn_inference: number;
    synthesizer: number;
    validator: number;
    total: number;
  };
}

export interface ChatResponse {
  message: string;
  dataset_queried: string;
  mode: string;
  intent: string;
  results: Record<string, unknown>[];
  insight: string;
  datasets_cited: string[];
  graph_context_summary: string;
  pipeline_timing_ms?: Record<string, number>;
  session_id?: string;
}

export interface InsertResult {
  ok: boolean;
  operation: string;
  nodes_created: number;
  edges_created: number;
  cypher_summary: string;
  detail?: Record<string, unknown>;
  error?: string;
  requires_confirm: boolean;
  preview?: Record<string, unknown>;
}

export interface DatasetStatus {
  name: string;
  on_disk: boolean;
  files: Record<string, boolean>;
  last_ingest?: string;
  neo4j_counts?: Record<string, number>;
}

export interface FriendRec {
  id: string;
  name: string;
  mutual_friends: number;
  influence_score: number;
  follower_count: number;
}

export interface TrendingPost {
  id: string;
  title: string;
  content: string;
  likes: number;
  comments: number;
  topic: string;
  created_at: string;
  engagement: number;
}

export interface Influencer {
  id: string;
  name: string;
  followers: number;
  gnn_score: number;
  post_count: number;
  avg_likes: number;
  composite_score: number;
}

export interface ConnectionPath {
  shortest_path: { node_names: string[]; rel_types: string[]; hops: number } | null;
  common_friends: { id: string; name: string }[];
  common_liked_posts: { id: string; title: string; topic: string }[];
}

export interface UserInfluenceStats {
  id: string;
  name: string;
  bio: string;
  followers: number;
  gnn_score: number;
  posts: number;
  friends: number;
  total_likes: number;
  avg_likes: number;
}

export interface LinkCandidate {
  id: string;
  name: string;
  graph_score: number;
}

export interface NLParseResponse {
  ok: boolean;
  dataset: string;
  operations: Record<string, unknown>[];
  parsed_names: string[];
  error?: string;
}

// ─── API Methods ──────────────────────────────────────────────────────────────

export const api = {
  // System
  health: () => request<HealthResponse>("/health"),
  vectorIndexes: () => request<{ indexes: unknown[] }>("/vector-indexes"),
  refreshEmbeddings: (force = false) =>
    request<{ status: string; counts: Record<string, number> }>(`/refresh-embeddings?force=${force}`, { method: "POST" }),
  gnnStatus: () =>
    request<{ loaded_datasets: string[]; all_configured_datasets: string[]; load_state: Record<string, boolean> }>("/gnn/status"),

  // Datasets
  datasetsStatus: () =>
    request<{ datasets: Record<string, DatasetStatus>; neo4j_connected: boolean }>("/datasets/status"),
  ingestDataset: (dataset?: string, force = false) =>
    request<{ triggered: string[]; results: Record<string, unknown> }>(
      `/datasets/ingest?${dataset ? `dataset=${dataset}&` : ""}force=${force}`,
      { method: "POST" }
    ),

  // Chat
  chat: (body: { message: string; dataset?: string; mode?: string; top_k?: number; session_id?: string; user_id?: string }) =>
    request<ChatResponse>("/chat", { method: "POST", body: JSON.stringify(body) }),

  // Insert
  insertParse: (nl_command: string, dataset?: string) =>
    request<NLParseResponse>("/chat/insert/parse", {
      method: "POST",
      body: JSON.stringify({ nl_command, dataset }),
    }),
  insertNL: (nl_command: string, dataset?: string, confirm = false) =>
    request<InsertResult>("/chat/insert", {
      method: "POST",
      body: JSON.stringify({ nl_command, dataset, confirm }),
    }),
  insertUser: (body: Record<string, unknown>, confirm = false) =>
    request<InsertResult>(`/chat/insert/user?confirm=${confirm}`, {
      method: "POST",
      body: JSON.stringify(body),
    }),
  insertEdge: (body: Record<string, unknown>, confirm = false) =>
    request<InsertResult>(`/chat/insert/edge?confirm=${confirm}`, {
      method: "POST",
      body: JSON.stringify(body),
    }),
  insertPost: (body: Record<string, unknown>, confirm = false) =>
    request<InsertResult>(`/chat/insert/post?confirm=${confirm}`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  // Pipeline
  query: (body: Record<string, unknown>) =>
    request<PipelineResult>("/query", { method: "POST", body: JSON.stringify(body) }),
  recommendFriends: (userId: string, topK = 10, dataset = "all") =>
    request<PipelineResult>(`/recommend-friends/${userId}?top_k=${topK}&dataset=${dataset}`),
  predictLinks: (body: Record<string, unknown>) =>
    request<PipelineResult>("/predict-links", { method: "POST", body: JSON.stringify(body) }),
  userInfluence: (userId: string, dataset = "all") =>
    request<PipelineResult>(`/user-influence/${userId}?dataset=${dataset}`),
  trendingPosts: (topK = 10, topic?: string, dataset = "all") =>
    request<PipelineResult>(`/trending-posts?top_k=${topK}${topic ? `&topic=${topic}` : ""}&dataset=${dataset}`),
  explainConnection: (userA: string, userB: string, dataset = "all") =>
    request<PipelineResult>(`/explain-connection?user_a=${userA}&user_b=${userB}&dataset=${dataset}`),

  // Direct graph
  graphFriendRecs: (userId: string, topK = 10) =>
    request<{ user_id: string; top_k: number; recommendations: FriendRec[] }>(
      `/graph/friend-recommendations/${userId}?top_k=${topK}`
    ),
  graphTrendingPosts: (topK = 10, topic?: string, hoursWindow = 48) =>
    request<{ top_k: number; topic?: string; hours_window: number; posts: TrendingPost[] }>(
      `/graph/trending-posts?top_k=${topK}&hours_window=${hoursWindow}${topic ? `&topic=${topic}` : ""}`
    ),
  graphUserStats: (userId: string) =>
    request<UserInfluenceStats>(`/graph/users/${userId}/influence-stats`),
  graphConnectionPath: (userA: string, userB: string) =>
    request<ConnectionPath>(`/graph/connection-path?user_a=${userA}&user_b=${userB}`),
  graphLinkCandidates: (userId: string, topK = 20) =>
    request<{ user_id: string; top_k: number; candidates: LinkCandidate[] }>(
      `/graph/link-prediction/candidates/${userId}?top_k=${topK}`
    ),
  graphTopInfluencers: (topK = 20) =>
    request<{ top_k: number; influencers: Influencer[] }>(`/graph/top-influencers?top_k=${topK}`),
};
