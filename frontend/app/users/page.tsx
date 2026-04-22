"use client";
import { useState } from "react";
import { Search, User, Loader2, Star, FileText, Users, Heart } from "lucide-react";
import { api, UserInfluenceStats } from "@/lib/api";
import { Panel, Badge, Button, Input, Select, SectionHeader, StatCard } from "@/components/ui";

function RoleClassification({ score }: { score: number }) {
  const role = score >= 0.8 ? "influencer" : score >= 0.6 ? "content_creator" : score >= 0.4 ? "community_hub" : "regular_user";
  const colors: Record<string, string> = { influencer: "cyan", content_creator: "purple", community_hub: "amber", regular_user: "green" };
  return <Badge variant={colors[role]}>{role.replace("_", " ")}</Badge>;
}

export default function UsersPage() {
  const [userId, setUserId] = useState("user_1");
  const [dataset, setDataset] = useState("all");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<UserInfluenceStats | null>(null);
  const [pipelineResult, setPipelineResult] = useState<Record<string, unknown> | null>(null);

  const lookupStats = async () => {
    setLoading(true); setError(null); setPipelineResult(null);
    try {
      const r = await api.graphUserStats(userId);
      setStats(r);
    } catch (e) { setError(String(e)); } finally { setLoading(false); }
  };

  const runInfluence = async () => {
    setLoading(true); setError(null); setStats(null);
    try {
      const r = await api.userInfluence(userId, dataset);
      setPipelineResult(r as unknown as Record<string, unknown>);
    } catch (e) { setError(String(e)); } finally { setLoading(false); }
  };

  return (
    <div className="p-6 max-w-5xl">
      <div className="mb-6 pt-2">
        <span className="text-xs font-mono px-2 py-1 rounded mb-2 inline-block" style={{ background: "rgba(168,85,247,0.1)", color: "#a855f7" }}>USERS</span>
        <h1 className="text-2xl font-bold font-display" style={{ color: "#e6edf3", letterSpacing: "-0.03em" }}>User Intelligence</h1>
        <p className="text-xs font-mono mt-0.5" style={{ color: "#484f58" }}>Influence stats and GNN-predicted user roles</p>
      </div>

      <Panel className="p-5 mb-6">
        <SectionHeader title="Lookup User" />
        <div className="flex items-center gap-3 flex-wrap">
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono" style={{ color: "#484f58" }}>User ID</span>
            <Input value={userId} onChange={setUserId} placeholder="user_1" className="w-36" />
          </div>
          <Select value={dataset} onChange={setDataset} options={[
            { value: "all", label: "All" }, { value: "facebook", label: "Facebook" },
            { value: "twitter", label: "Twitter" }, { value: "reddit", label: "Reddit" }, { value: "demo", label: "Demo" },
          ]} />
          <Button onClick={lookupStats} loading={loading} variant="primary">
            <Search className="w-3.5 h-3.5" /> Graph Stats
          </Button>
          <Button onClick={runInfluence} loading={loading} variant="secondary">
            <Star className="w-3.5 h-3.5" /> GNN Influence
          </Button>
        </div>
      </Panel>

      {error && (
        <div className="mb-4 p-3 rounded-lg text-xs font-mono" style={{ background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.2)", color: "#ef4444" }}>
          {error}
        </div>
      )}

      {loading && (
        <div className="flex items-center justify-center py-16">
          <Loader2 className="w-5 h-5 animate-spin" style={{ color: "#00d4ff" }} />
        </div>
      )}

      {stats && !loading && (
        <div className="space-y-5">
          <Panel className="p-5">
            <div className="flex items-start gap-4">
              <div className="w-12 h-12 rounded-xl flex items-center justify-center flex-shrink-0"
                style={{ background: "rgba(0,212,255,0.1)", border: "1px solid rgba(0,212,255,0.2)" }}>
                <User className="w-5 h-5" style={{ color: "#00d4ff" }} />
              </div>
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-1 flex-wrap">
                  <h2 className="text-lg font-bold font-display" style={{ color: "#e6edf3" }}>{stats.name}</h2>
                  <RoleClassification score={stats.gnn_score || 0} />
                </div>
                <p className="text-xs font-mono mb-1" style={{ color: "#484f58" }}>{stats.id}</p>
                {stats.bio && <p className="text-sm" style={{ color: "#8b949e" }}>{stats.bio}</p>}
              </div>
              <div className="text-right">
                <div className="text-2xl font-bold font-display" style={{ color: "#a855f7", letterSpacing: "-0.04em" }}>
                  {(stats.gnn_score || 0).toFixed(2)}
                </div>
                <div className="text-[10px] font-mono" style={{ color: "#484f58" }}>GNN Score</div>
              </div>
            </div>
          </Panel>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <StatCard label="Followers" value={stats.followers?.toLocaleString() || "0"} color="#00d4ff" icon={<Users className="w-4 h-4" />} />
            <StatCard label="Posts" value={stats.posts || 0} color="#a855f7" icon={<FileText className="w-4 h-4" />} />
            <StatCard label="Friends" value={stats.friends || 0} color="#10b981" icon={<Users className="w-4 h-4" />} />
            <StatCard label="Total Likes" value={stats.total_likes?.toLocaleString() || "0"} sub={`avg ${(stats.avg_likes || 0).toFixed(1)}/post`} color="#fbbf24" icon={<Heart className="w-4 h-4" />} />
          </div>
        </div>
      )}

      {pipelineResult && !loading && (
        <Panel className="p-5">
          <SectionHeader title="GNN Influence Analysis" sub="Multi-agent pipeline result" />
          <div className="space-y-3">
            {(pipelineResult.results as Record<string, unknown>[])?.slice(0, 5).map((r, i) => (
              <div key={i} className="p-3 rounded-lg" style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.06)" }}>
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium" style={{ color: "#e6edf3" }}>{String(r.name || r.id || `Result ${i + 1}`)}</span>
                  <div className="flex gap-2">
                    {typeof r.gnn_score === "number" && <Badge variant="purple">GNN: {(r.gnn_score as number).toFixed(2)}</Badge>}
                    {typeof r.influence_score === "number" && <Badge variant="cyan">{(r.influence_score as number).toFixed(2)}</Badge>}
                  </div>
                </div>
              </div>
            ))}
            {pipelineResult.insight != null && (
              <div className="mt-3 p-4 rounded-xl" style={{ background: "rgba(0,212,255,0.05)", border: "1px solid rgba(0,212,255,0.1)" }}>
                <div className="text-[10px] font-mono mb-1.5" style={{ color: "#00d4ff" }}>AI INSIGHT</div>
                <p className="text-sm" style={{ color: "#8b949e" }}>{String(pipelineResult.insight)}</p>
              </div>
            )}
          </div>
        </Panel>
      )}
    </div>
  );
}
