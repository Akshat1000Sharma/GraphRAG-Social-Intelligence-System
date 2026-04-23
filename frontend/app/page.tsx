"use client";
import { useEffect, useState } from "react";
import { Activity, Database, Cpu, Users, RefreshCw } from "lucide-react";
import { api, HealthResponse, GnnStatusResponse } from "@/lib/api";
import { StatCard, Panel, Badge, Button, Skeleton, StatusDot, SectionHeader } from "@/components/ui";

/** Display order for GNN platforms (matches backend DATASET_CONFIG). */
const GNN_PLATFORMS = ["facebook", "twitter", "reddit"] as const;

function badgeVariantForDataset(name: string): "cyan" | "purple" | "amber" | "green" {
  if (name === "facebook") return "cyan";
  if (name === "twitter") return "purple";
  if (name === "reddit") return "amber";
  return "green";
}

function PipelineStep({ label, done }: { label: string; done?: boolean }) {
  return (
    <div className="flex items-center gap-3">
      <div className="w-2 h-2 rounded-full flex-shrink-0" style={{
        background: done ? "#10b981" : "#21262d",
        boxShadow: done ? "0 0 6px rgba(16,185,129,0.7)" : "none",
      }} />
      <span className="text-xs font-mono" style={{ color: done ? "#10b981" : "#484f58" }}>{label}</span>
    </div>
  );
}

export default function Dashboard() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [gnnStatus, setGnnStatus] = useState<GnnStatusResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    setLoading(true); setError(null);
    try {
      const [h, g] = await Promise.all([api.health(), api.gnnStatus().catch(() => null)]);
      setHealth(h);
      setGnnStatus(g);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, []);

  const totalUsers = health?.dataset_counts ? Object.values(health.dataset_counts).reduce((s, d) => s + (d.users || 0), 0) : 0;
  const totalPosts = health?.dataset_counts ? Object.values(health.dataset_counts).reduce((s, d) => s + (d.posts || 0), 0) : 0;

  const loadedSet = new Set(gnnStatus?.loaded_datasets ?? health?.gnn_datasets ?? []);
  const gnnLoadedCount = GNN_PLATFORMS.filter((p) => loadedSet.has(p)).length;
  const gnnTotalPlatforms = GNN_PLATFORMS.length;

  return (
    <div className="p-6 max-w-6xl">
      <div className="mb-8 pt-2">
        <div className="flex items-center gap-3 mb-2">
          <span className="text-xs font-mono px-2 py-1 rounded" style={{ background: "rgba(0,212,255,0.1)", color: "#00d4ff" }}>SYSTEM DASHBOARD</span>
          {!loading && health && (
            <div className="flex items-center gap-1.5">
              <StatusDot ok={health.status === "healthy"} />
              <span className="text-xs font-mono" style={{ color: health.status === "healthy" ? "#10b981" : "#ef4444" }}>{health.status?.toUpperCase()}</span>
            </div>
          )}
        </div>
        <h1 className="text-3xl font-bold font-display glow-text" style={{ color: "#e6edf3", letterSpacing: "-0.04em" }}>Graph Intelligence</h1>
        <p className="text-sm font-mono mt-1" style={{ color: "#484f58" }}>GNN · GraphRAG · Multi-Agent KAG Pipeline</p>
      </div>

      {error && (
        <div className="mb-6 p-4 rounded-xl text-sm font-mono" style={{ background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.2)", color: "#ef4444" }}>
          ⚠ API unreachable — {error} <button onClick={load} className="ml-3 underline">retry</button>
        </div>
      )}

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        {loading ? Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-24" />) : (<>
          <StatCard label="Neo4j" value={health?.neo4j_connected ? "Online" : "Offline"} sub="Graph database" color={health?.neo4j_connected ? "#10b981" : "#ef4444"} icon={<Database className="w-4 h-4" />} />
          <StatCard
            label="GNN Models"
            value={`${gnnLoadedCount}/${gnnTotalPlatforms}`}
            sub="Facebook · Twitter · Reddit"
            color={gnnLoadedCount > 0 ? "#a855f7" : "#484f58"}
            icon={<Cpu className="w-4 h-4" />}
          />
          <StatCard label="Total Users" value={totalUsers.toLocaleString()} sub="Across all datasets" color="#00d4ff" icon={<Users className="w-4 h-4" />} />
          <StatCard label="Total Posts" value={totalPosts.toLocaleString()} sub="Indexed content" color="#fbbf24" icon={<Activity className="w-4 h-4" />} />
        </>)}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Panel className="p-5">
          <SectionHeader title="System Status" sub={`v${health?.version || "—"}`}
            action={<Button variant="ghost" size="sm" onClick={load} loading={loading}><RefreshCw className="w-3 h-3" /></Button>} />
          <div className="space-y-3">
            {[{ k: "Neo4j Database", v: health?.neo4j_connected }, { k: "GNN Engine (any)", v: health?.gnn_loaded || gnnLoadedCount > 0 }, { k: "Pipeline", v: health?.pipeline_ready }, { k: "Vector Backend", v: !!health?.vector_backend }].map(({ k, v }) => (
              <div key={k} className="flex items-center justify-between">
                <span className="text-xs font-mono" style={{ color: "#8b949e" }}>{k}</span>
                <div className="flex items-center gap-2">
                  <StatusDot ok={!!v} loading={loading} />
                  <span className="text-xs font-mono" style={{ color: loading ? "#484f58" : v ? "#10b981" : "#ef4444" }}>{loading ? "…" : v ? "OK" : "FAIL"}</span>
                </div>
              </div>
            ))}
          </div>
          {health?.vector_backend && (
            <div className="mt-4 pt-4 border-t" style={{ borderColor: "rgba(255,255,255,0.06)" }}>
              <div className="text-[11px] font-mono mb-1" style={{ color: "#484f58" }}>Vector Backend</div>
              <Badge variant="cyan">{health.vector_backend}</Badge>
            </div>
          )}
        </Panel>

        <Panel className="p-5">
          <SectionHeader title="Datasets" sub="Ingested graph data" />
          {loading ? <div className="space-y-2">{Array.from({ length: 3 }).map((_, i) => <Skeleton key={i} className="h-10" />)}</div>
          : health?.dataset_counts && Object.keys(health.dataset_counts).length > 0
          ? <div className="space-y-3">
              {Object.entries(health.dataset_counts).map(([name, counts]) => (
                <div key={name} className="p-3 rounded-lg" style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.06)" }}>
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs font-mono font-semibold" style={{ color: "#e6edf3" }}>{name}</span>
                    <Badge variant={badgeVariantForDataset(name)}>{name}</Badge>
                  </div>
                  <div className="flex gap-3">
                    {Object.entries(counts).map(([k, v]) => (
                      <span key={k} className="text-[11px] font-mono" style={{ color: "#484f58" }}>
                        {k}: <span style={{ color: "#8b949e" }}>{v}</span>
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          : <p className="text-xs font-mono" style={{ color: "#484f58" }}>No dataset data available</p>}
        </Panel>

        <Panel className="p-5">
          <SectionHeader title="Pipeline" sub="Multi-agent stages" />
          <div className="space-y-2.5">
            {[
              { label: "QueryAnalyzer → intent, entities", done: true },
              { label: "RouterAgent → query_type", done: true },
              { label: "GraphRetriever (Cypher)", done: health?.neo4j_connected },
              { label: "VectorRetriever (Neo4j ANN)", done: health?.neo4j_connected },
              { label: "GNN Inference (CPU)", done: health?.gnn_loaded },
              { label: "Synthesizer (KAG + LLM)", done: health?.pipeline_ready },
              { label: "Validator + dedup", done: health?.pipeline_ready },
            ].map(({ label, done }, i) => <PipelineStep key={i} label={label} done={!!done} />)}
          </div>
          <div className="mt-4 pt-4 border-t" style={{ borderColor: "rgba(255,255,255,0.06)" }}>
            <div className="text-[11px] font-mono mb-2" style={{ color: "#484f58" }}>GNN by platform (weights in /weights)</div>
            <div className="space-y-2">
              {GNN_PLATFORMS.map((name) => {
                const ok = loadedSet.has(name);
                return (
                  <div key={name} className="flex items-center justify-between gap-2">
                    <div className="flex items-center gap-2 min-w-0">
                      <StatusDot ok={ok} loading={loading} />
                      <Badge variant={badgeVariantForDataset(name)}>{name}</Badge>
                    </div>
                    <span className="text-[10px] font-mono truncate" style={{ color: ok ? "#10b981" : "#6e7681" }}>
                      {ok ? "loaded" : "not in memory"}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        </Panel>
      </div>

      {health?.vector_indexes && health.vector_indexes.length > 0 && (
        <Panel className="p-5 mt-6">
          <SectionHeader title="Vector Indexes" sub="Neo4j HNSW indexes" />
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {health.vector_indexes.map((idx) => (
              <div key={idx.name} className="p-3 rounded-lg" style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.06)" }}>
                <div className="text-xs font-mono mb-1" style={{ color: "#e6edf3" }}>{idx.name}</div>
                <Badge variant={idx.state === "ONLINE" ? "green" : "amber"}>{idx.state}</Badge>
              </div>
            ))}
          </div>
        </Panel>
      )}
    </div>
  );
}
