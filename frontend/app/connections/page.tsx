"use client";
import { useState } from "react";
import { Search, GitBranch, ArrowRight, Loader2 } from "lucide-react";
import { api, ConnectionPath } from "@/lib/api";
import { GNN_MODEL_OPTIONS } from "@/lib/dataset-options";
import { Panel, Badge, Button, Input, Select, SectionHeader, Empty } from "@/components/ui";

function PathViz({ path }: { path: NonNullable<ConnectionPath["shortest_path"]> }) {
  return (
    <div className="py-4 overflow-x-auto">
      <div className="flex items-center gap-1 min-w-max">
        {path.node_names.map((name, i) => (
          <div key={i} className="flex items-center gap-1">
            <div
              className="px-3 py-2 rounded-lg text-xs font-mono text-center"
              style={{
                background: i === 0 ? "rgba(0,212,255,0.12)" : i === path.node_names.length - 1 ? "rgba(168,85,247,0.12)" : "rgba(255,255,255,0.05)",
                border: `1px solid ${i === 0 ? "rgba(0,212,255,0.3)" : i === path.node_names.length - 1 ? "rgba(168,85,247,0.3)" : "rgba(255,255,255,0.1)"}`,
                color: i === 0 ? "#00d4ff" : i === path.node_names.length - 1 ? "#a855f7" : "#e6edf3",
                minWidth: "80px",
              }}
            >
              {name}
            </div>
            {i < path.rel_types.length && (
              <div className="flex flex-col items-center gap-0.5 px-2">
                <span className="text-[9px] font-mono" style={{ color: "#484f58" }}>{path.rel_types[i]}</span>
                <ArrowRight className="w-3 h-3" style={{ color: "#484f58" }} />
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

export default function ConnectionsPage() {
  const [userA, setUserA] = useState("user_1");
  const [userB, setUserB] = useState("user_5");
  const [dataset, setDataset] = useState("all");
  const [gnnDataset, setGnnDataset] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pathData, setPathData] = useState<ConnectionPath | null>(null);
  const [explainData, setExplainData] = useState<Record<string, unknown> | null>(null);

  const queryPath = async () => {
    setLoading(true); setError(null);
    try {
      const r = await api.graphConnectionPath(userA, userB);
      setPathData(r);
    } catch (e) { setError(String(e)); } finally { setLoading(false); }
  };

  const explainConnection = async () => {
    setLoading(true); setError(null);
    try {
      const r = await api.explainConnection(userA, userB, dataset, gnnDataset || undefined);
      setExplainData(r as unknown as Record<string, unknown>);
    } catch (e) { setError(String(e)); } finally { setLoading(false); }
  };

  return (
    <div className="p-6 max-w-5xl">
      <div className="mb-6 pt-2">
        <span className="text-xs font-mono px-2 py-1 rounded mb-2 inline-block" style={{ background: "rgba(16,185,129,0.1)", color: "#10b981" }}>CONNECTIONS</span>
        <h1 className="text-2xl font-bold font-display" style={{ color: "#e6edf3", letterSpacing: "-0.03em" }}>Connection Explorer</h1>
        <p className="text-xs font-mono mt-0.5" style={{ color: "#484f58" }}>Shortest paths, common friends, LLM-generated explanations</p>
      </div>

      <Panel className="p-5 mb-6">
        <SectionHeader title="Find Connection" />
        <div className="flex items-center gap-3 flex-wrap">
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono" style={{ color: "#00d4ff" }}>From</span>
            <Input value={userA} onChange={setUserA} placeholder="user_1" className="w-28" />
          </div>
          <GitBranch className="w-4 h-4" style={{ color: "#484f58" }} />
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono" style={{ color: "#a855f7" }}>To</span>
            <Input value={userB} onChange={setUserB} placeholder="user_5" className="w-28" />
          </div>
          <Select value={dataset} onChange={setDataset} options={[
            { value: "all", label: "All" }, { value: "facebook", label: "Facebook" },
            { value: "twitter", label: "Twitter" }, { value: "reddit", label: "Reddit" }, { value: "demo", label: "Demo" },
          ]} />
          <Select value={gnnDataset} onChange={setGnnDataset} options={GNN_MODEL_OPTIONS} />
          <Button onClick={queryPath} loading={loading} variant="primary">
            <Search className="w-3.5 h-3.5" /> Find Path
          </Button>
          <Button onClick={explainConnection} loading={loading} variant="secondary">
            <GitBranch className="w-3.5 h-3.5" /> AI Explain
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

      {pathData && !loading && (
        <div className="space-y-5">
          {/* Shortest path */}
          <Panel className="p-5">
            <SectionHeader title="Shortest Path" sub={pathData.shortest_path ? `${pathData.shortest_path.hops} hop${pathData.shortest_path.hops !== 1 ? "s" : ""}` : "No path found"} />
            {pathData.shortest_path ? (
              <>
                <PathViz path={pathData.shortest_path} />
                <div className="flex items-center gap-2 mt-2">
                  <Badge variant="green">{pathData.shortest_path.hops} hops</Badge>
                  {pathData.shortest_path.rel_types.map((r, i) => <Badge key={i} variant="ghost">{r}</Badge>)}
                </div>
              </>
            ) : <Empty message="No path found between these users" />}
          </Panel>

          {/* Common friends */}
          {pathData.common_friends && pathData.common_friends.length > 0 && (
            <Panel className="p-5">
              <SectionHeader title="Common Friends" sub={`${pathData.common_friends.length} mutual`} />
              <div className="flex flex-wrap gap-2">
                {pathData.common_friends.map((f) => (
                  <div key={f.id} className="px-3 py-1.5 rounded-lg text-xs font-mono"
                    style={{ background: "rgba(0,212,255,0.08)", border: "1px solid rgba(0,212,255,0.15)", color: "#00d4ff" }}>
                    {f.name}
                  </div>
                ))}
              </div>
            </Panel>
          )}

          {/* Common liked posts */}
          {pathData.common_liked_posts && pathData.common_liked_posts.length > 0 && (
            <Panel className="p-5">
              <SectionHeader title="Common Liked Posts" sub={`${pathData.common_liked_posts.length} shared`} />
              <div className="space-y-2">
                {pathData.common_liked_posts.map((p) => (
                  <div key={p.id} className="flex items-center gap-3 p-2.5 rounded-lg"
                    style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.06)" }}>
                    <div className="flex-1 text-xs font-medium" style={{ color: "#e6edf3" }}>{p.title}</div>
                    <Badge variant="purple">{p.topic}</Badge>
                  </div>
                ))}
              </div>
            </Panel>
          )}
        </div>
      )}

      {explainData && !loading && (
        <Panel className="p-5">
          <SectionHeader title="AI-Generated Explanation" sub="Multi-agent pipeline + LLM" />
          {explainData.insight != null && (
            <div className="p-4 rounded-xl mb-4" style={{ background: "rgba(0,212,255,0.05)", border: "1px solid rgba(0,212,255,0.1)" }}>
              <div className="text-[10px] font-mono mb-1.5" style={{ color: "#00d4ff" }}>INSIGHT</div>
              <p className="text-sm leading-relaxed" style={{ color: "#8b949e" }}>{String(explainData.insight)}</p>
            </div>
          )}
          {Array.isArray(explainData.results) && explainData.results.length > 0 && (
            <div className="space-y-2">
              {(explainData.results as Record<string, unknown>[]).slice(0, 5).map((r, i) => (
                <div key={i} className="p-2.5 rounded-lg text-xs font-mono"
                  style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.06)", color: "#8b949e" }}>
                  {JSON.stringify(r)}
                </div>
              ))}
            </div>
          )}
        </Panel>
      )}
    </div>
  );
}
