"use client";
import { useState } from "react";
import { TrendingUp, Flame, BarChart2, Loader2, Search } from "lucide-react";
import { api, TrendingPost } from "@/lib/api";
import { Panel, Badge, Button, Input, Select, SectionHeader, Empty, StatCard } from "@/components/ui";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";

function PostCard({ post, rank }: { post: TrendingPost; rank: number }) {
  const age = post.created_at
    ? Math.floor((Date.now() - new Date(post.created_at).getTime()) / 3600000) + "h ago"
    : "—";
  return (
    <div className="p-4 rounded-xl card-hover"
      style={{ background: "rgba(255,255,255,0.025)", border: "1px solid rgba(255,255,255,0.06)" }}>
      <div className="flex items-start justify-between gap-3 mb-2">
        <div className="flex items-center gap-2">
          <span className="text-lg font-bold font-mono" style={{ color: rank <= 3 ? "#fbbf24" : "#484f58" }}>
            {rank <= 3 ? (rank === 1 ? "🔥" : rank === 2 ? "⚡" : "✨") : `#${rank}`}
          </span>
          <Badge variant={post.topic === "AI" ? "cyan" : post.topic === "tech" ? "purple" : post.topic === "news" ? "amber" : "green"}>
            {post.topic}
          </Badge>
        </div>
        <span className="text-[10px] font-mono" style={{ color: "#484f58" }}>{age}</span>
      </div>
      <h3 className="text-sm font-medium mb-1.5" style={{ color: "#e6edf3" }}>{post.title}</h3>
      {post.content && (
        <p className="text-xs mb-2 line-clamp-2" style={{ color: "#8b949e" }}>{post.content}</p>
      )}
      <div className="flex items-center gap-4">
        <span className="text-xs font-mono" style={{ color: "#10b981" }}>
          ♥ {post.likes?.toLocaleString() || 0}
        </span>
        <span className="text-xs font-mono" style={{ color: "#00d4ff" }}>
          💬 {post.comments?.toLocaleString() || 0}
        </span>
        <span className="text-xs font-mono ml-auto" style={{ color: "#fbbf24" }}>
          ⚡ {post.engagement?.toFixed(0) || 0}
        </span>
      </div>
    </div>
  );
}

export default function AnalyticsPage() {
  const [posts, setPosts] = useState<TrendingPost[] | null>(null);
  const [topic, setTopic] = useState("");
  const [topK, setTopK] = useState("10");
  const [dataset, setDataset] = useState("all");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    setLoading(true); setError(null);
    try {
      const r = await api.graphTrendingPosts(parseInt(topK), topic || undefined, 48, dataset);
      setPosts(r.posts);
    } catch (e) { setError(String(e)); } finally { setLoading(false); }
  };

  const chartData = posts?.slice(0, 10).map((p) => ({
    name: p.title?.slice(0, 18) + "…",
    engagement: p.engagement,
    likes: p.likes,
    comments: p.comments,
  })) || [];

  return (
    <div className="p-6 max-w-6xl">
      <div className="mb-6 pt-2">
        <span className="text-xs font-mono px-2 py-1 rounded mb-2 inline-block" style={{ background: "rgba(251,191,36,0.1)", color: "#fbbf24" }}>ANALYTICS</span>
        <h1 className="text-2xl font-bold font-display" style={{ color: "#e6edf3", letterSpacing: "-0.03em" }}>Trending Content</h1>
        <p className="text-xs font-mono mt-0.5" style={{ color: "#484f58" }}>Posts ranked by engagement velocity</p>
      </div>

      {/* Controls */}
      <div className="flex items-center gap-3 mb-5 flex-wrap">
        <Input value={topic} onChange={setTopic} placeholder="Filter by topic (AI, tech, …)" className="w-52" />
        <Select value={topK} onChange={setTopK} options={[
          { value: "5", label: "Top 5" }, { value: "10", label: "Top 10" },
          { value: "20", label: "Top 20" }, { value: "50", label: "Top 50" },
        ]} />
        <Select value={dataset} onChange={setDataset} options={[
          { value: "all", label: "All" }, { value: "facebook", label: "Facebook" },
          { value: "twitter", label: "Twitter" }, { value: "reddit", label: "Reddit" },
        ]} />
        <Button onClick={load} loading={loading}>
          <Search className="w-3.5 h-3.5" /> Load Trending
        </Button>
      </div>

      {error && (
        <div className="mb-4 p-3 rounded-lg text-xs font-mono" style={{ background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.2)", color: "#ef4444" }}>
          {error}
        </div>
      )}

      {loading && (
        <div className="flex items-center justify-center py-16">
          <Loader2 className="w-6 h-6 animate-spin" style={{ color: "#00d4ff" }} />
        </div>
      )}

      {posts && !loading && (
        <>
          {/* Stats */}
          <div className="grid grid-cols-3 gap-4 mb-6">
            <StatCard label="Posts Found" value={posts.length} color="#fbbf24" icon={<TrendingUp className="w-4 h-4" />} />
            <StatCard
              label="Avg Engagement"
              value={(posts.reduce((s, p) => s + (p.engagement || 0), 0) / posts.length).toFixed(0)}
              color="#00d4ff"
              icon={<BarChart2 className="w-4 h-4" />}
            />
            <StatCard
              label="Top Score"
              value={(posts[0]?.engagement || 0).toFixed(0)}
              sub={posts[0]?.title?.slice(0, 30) + "…"}
              color="#10b981"
              icon={<Flame className="w-4 h-4" />}
            />
          </div>

          {/* Chart */}
          {chartData.length > 0 && (
            <Panel className="p-5 mb-6">
              <SectionHeader title="Engagement Chart" sub="Likes + 2×comments / age" />
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
                    <XAxis dataKey="name" tick={{ fill: "#484f58", fontSize: 9, fontFamily: "IBM Plex Mono" }} />
                    <YAxis tick={{ fill: "#484f58", fontSize: 9, fontFamily: "IBM Plex Mono" }} />
                    <Tooltip
                      contentStyle={{ background: "#0d1117", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 8, fontSize: 11, fontFamily: "IBM Plex Mono" }}
                      labelStyle={{ color: "#e6edf3" }}
                      itemStyle={{ color: "#00d4ff" }}
                    />
                    <Bar dataKey="engagement" radius={[3, 3, 0, 0]}>
                      {chartData.map((_, i) => (
                        <Cell key={i} fill={i < 3 ? "#fbbf24" : "#00d4ff"} fillOpacity={0.7} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Panel>
          )}

          {/* Grid */}
          {posts.length === 0 ? <Empty message="No posts found" /> : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {posts.map((p, i) => <PostCard key={p.id} post={p} rank={i + 1} />)}
            </div>
          )}
        </>
      )}

      {!posts && !loading && (
        <div className="flex flex-col items-center justify-center py-16 text-center">
          <TrendingUp className="w-10 h-10 mb-3" style={{ color: "#21262d" }} />
          <p className="text-sm font-mono" style={{ color: "#484f58" }}>Click "Load Trending" to fetch posts</p>
        </div>
      )}
    </div>
  );
}
