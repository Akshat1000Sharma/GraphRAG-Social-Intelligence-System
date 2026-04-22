"use client";
import { useState } from "react";
import { Search, Users, Star, GitBranch, Loader2 } from "lucide-react";
import { api, FriendRec, Influencer, LinkCandidate } from "@/lib/api";
import { Panel, Badge, Button, Input, Select, Tabs, SectionHeader, Empty, StatCard } from "@/components/ui";

function ScoreBar({ value, max = 1, color = "#00d4ff" }: { value: number; max?: number; color?: string }) {
  const pct = Math.min((value / max) * 100, 100);
  return (
    <div className="flex items-center gap-2">
      <div className="h-1.5 rounded-full flex-1 overflow-hidden" style={{ background: "rgba(255,255,255,0.06)" }}>
        <div className="h-full rounded-full" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="text-[10px] font-mono w-8 text-right" style={{ color: "#8b949e" }}>{value.toFixed ? value.toFixed(2) : value}</span>
    </div>
  );
}

export default function ExplorePage() {
  const [tab, setTab] = useState("Friends");
  const [userId, setUserId] = useState("user_1");
  const [dataset, setDataset] = useState("all");
  const [topK, setTopK] = useState("10");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [friends, setFriends] = useState<FriendRec[] | null>(null);
  const [influencers, setInfluencers] = useState<Influencer[] | null>(null);
  const [candidates, setCandidates] = useState<LinkCandidate[] | null>(null);

  const datasetOpts = [
    { value: "all", label: "All" },
    { value: "facebook", label: "Facebook" },
    { value: "twitter", label: "Twitter" },
    { value: "reddit", label: "Reddit" },
    { value: "demo", label: "Demo" },
  ];

  const run = async () => {
    setLoading(true); setError(null);
    try {
      if (tab === "Friends") {
        const r = await api.graphFriendRecs(userId, parseInt(topK));
        setFriends(r.recommendations);
      } else if (tab === "Influencers") {
        const r = await api.graphTopInfluencers(parseInt(topK));
        setInfluencers(r.influencers);
      } else if (tab === "Link Candidates") {
        const r = await api.graphLinkCandidates(userId, parseInt(topK));
        setCandidates(r.candidates);
      }
    } catch (e) { setError(String(e)); } finally { setLoading(false); }
  };

  return (
    <div className="p-6 max-w-6xl">
      <div className="mb-6 pt-2">
        <span className="text-xs font-mono px-2 py-1 rounded mb-2 inline-block" style={{ background: "rgba(0,212,255,0.1)", color: "#00d4ff" }}>GRAPH EXPLORER</span>
        <h1 className="text-2xl font-bold font-display" style={{ color: "#e6edf3", letterSpacing: "-0.03em" }}>Explore Graph</h1>
        <p className="text-xs font-mono mt-0.5" style={{ color: "#484f58" }}>Direct Neo4j queries — no LLM synthesis</p>
      </div>

      <div className="flex items-center gap-3 mb-5 flex-wrap">
        <Tabs tabs={["Friends", "Influencers", "Link Candidates"]} active={tab} onChange={setTab} />
      </div>

      <div className="flex items-center gap-3 mb-5 flex-wrap">
        {tab !== "Influencers" && (
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono" style={{ color: "#484f58" }}>User ID</span>
            <Input value={userId} onChange={setUserId} placeholder="user_1" className="w-28" />
          </div>
        )}
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono" style={{ color: "#484f58" }}>Top K</span>
          <Select value={topK} onChange={setTopK} options={[
            { value: "5", label: "5" }, { value: "10", label: "10" },
            { value: "20", label: "20" }, { value: "50", label: "50" },
          ]} />
        </div>
        {tab !== "Influencers" && (
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono" style={{ color: "#484f58" }}>Dataset</span>
            <Select value={dataset} onChange={setDataset} options={datasetOpts} />
          </div>
        )}
        <Button onClick={run} loading={loading}>
          <Search className="w-3.5 h-3.5" /> Query
        </Button>
      </div>

      {error && (
        <div className="mb-4 p-3 rounded-lg text-xs font-mono" style={{ background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.2)", color: "#ef4444" }}>
          {error}
        </div>
      )}

      {/* Friend Recommendations */}
      {tab === "Friends" && (
        <Panel className="p-5">
          <SectionHeader title="Friend Recommendations" sub={friends ? `${friends.length} results for ${userId}` : "Run a query to see results"} />
          {loading && <div className="flex items-center gap-2 py-8 justify-center"><Loader2 className="w-4 h-4 animate-spin" style={{ color: "#00d4ff" }} /></div>}
          {!loading && friends && friends.length === 0 && <Empty message="No recommendations found" />}
          {!loading && friends && friends.length > 0 && (
            <div className="space-y-2">
              {friends.map((f, i) => (
                <div key={f.id} className="flex items-center gap-4 p-3 rounded-lg transition-all"
                  style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.05)" }}>
                  <span className="text-xs font-mono w-5 text-center" style={{ color: "#484f58" }}>#{i + 1}</span>
                  <div className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0"
                    style={{ background: "rgba(0,212,255,0.1)" }}>
                    <Users className="w-3.5 h-3.5" style={{ color: "#00d4ff" }} />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-medium" style={{ color: "#e6edf3" }}>{f.name}</div>
                    <div className="text-[10px] font-mono" style={{ color: "#484f58" }}>{f.id}</div>
                  </div>
                  <div className="flex items-center gap-1 text-xs font-mono" style={{ color: "#8b949e" }}>
                    <GitBranch className="w-3 h-3" /> {f.mutual_friends}
                  </div>
                  <div className="w-32">
                    <div className="text-[10px] font-mono mb-0.5" style={{ color: "#484f58" }}>influence</div>
                    <ScoreBar value={f.influence_score} color="#00d4ff" />
                  </div>
                  <div className="text-right">
                    <div className="text-xs font-mono" style={{ color: "#8b949e" }}>{f.follower_count?.toLocaleString()}</div>
                    <div className="text-[10px] font-mono" style={{ color: "#484f58" }}>followers</div>
                  </div>
                </div>
              ))}
            </div>
          )}
          {!loading && !friends && (
            <Empty message="Enter a user ID and click Query" />
          )}
        </Panel>
      )}

      {/* Top Influencers */}
      {tab === "Influencers" && (
        <Panel className="p-5">
          <SectionHeader title="Top Influencers" sub={influencers ? `${influencers.length} leaders by composite score` : "Run a query to see results"} />
          {loading && <div className="flex items-center gap-2 py-8 justify-center"><Loader2 className="w-4 h-4 animate-spin" style={{ color: "#00d4ff" }} /></div>}
          {!loading && influencers && influencers.length === 0 && <Empty message="No influencers found" />}
          {!loading && influencers && influencers.length > 0 && (
            <>
              <div className="grid grid-cols-3 gap-3 mb-4">
                {influencers.slice(0, 3).map((inf, i) => (
                  <div key={inf.id} className="p-3 rounded-xl text-center"
                    style={{ background: i === 0 ? "rgba(251,191,36,0.08)" : "rgba(255,255,255,0.03)", border: `1px solid ${i === 0 ? "rgba(251,191,36,0.2)" : "rgba(255,255,255,0.06)"}` }}>
                    <div className="text-lg font-mono mb-1" style={{ color: i === 0 ? "#fbbf24" : i === 1 ? "#9ca3af" : "#b45309" }}>
                      {i === 0 ? "🥇" : i === 1 ? "🥈" : "🥉"}
                    </div>
                    <div className="text-xs font-medium" style={{ color: "#e6edf3" }}>{inf.name}</div>
                    <div className="text-[10px] font-mono mt-1" style={{ color: "#484f58" }}>{inf.composite_score?.toFixed(0)} score</div>
                  </div>
                ))}
              </div>
              <div className="space-y-2">
                {influencers.map((inf, i) => (
                  <div key={inf.id} className="flex items-center gap-4 p-3 rounded-lg"
                    style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.05)" }}>
                    <span className="text-xs font-mono w-5" style={{ color: "#484f58" }}>#{i + 1}</span>
                    <div className="flex-1">
                      <div className="text-sm font-medium" style={{ color: "#e6edf3" }}>{inf.name}</div>
                      <div className="text-[10px] font-mono" style={{ color: "#484f58" }}>{inf.id}</div>
                    </div>
                    <div className="grid grid-cols-3 gap-3 text-right">
                      <div>
                        <div className="text-xs font-mono" style={{ color: "#a855f7" }}>{inf.gnn_score?.toFixed(2)}</div>
                        <div className="text-[10px] font-mono" style={{ color: "#484f58" }}>GNN</div>
                      </div>
                      <div>
                        <div className="text-xs font-mono" style={{ color: "#00d4ff" }}>{inf.followers?.toLocaleString()}</div>
                        <div className="text-[10px] font-mono" style={{ color: "#484f58" }}>followers</div>
                      </div>
                      <div>
                        <div className="text-xs font-mono" style={{ color: "#fbbf24" }}>{inf.avg_likes?.toFixed(0)}</div>
                        <div className="text-[10px] font-mono" style={{ color: "#484f58" }}>avg likes</div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
          {!loading && !influencers && <Empty message="Click Query to load top influencers" />}
        </Panel>
      )}

      {/* Link Candidates */}
      {tab === "Link Candidates" && (
        <Panel className="p-5">
          <SectionHeader title="Link Prediction Candidates" sub={candidates ? `${candidates.length} candidates for ${userId}` : "Run a query to see results"} />
          {loading && <div className="flex items-center gap-2 py-8 justify-center"><Loader2 className="w-4 h-4 animate-spin" style={{ color: "#00d4ff" }} /></div>}
          {!loading && candidates && candidates.length === 0 && <Empty message="No candidates found" />}
          {!loading && candidates && candidates.length > 0 && (
            <div className="space-y-2">
              {candidates.map((c, i) => (
                <div key={c.id} className="flex items-center gap-4 p-3 rounded-lg"
                  style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.05)" }}>
                  <span className="text-xs font-mono w-5" style={{ color: "#484f58" }}>#{i + 1}</span>
                  <div className="flex-1">
                    <div className="text-sm font-medium" style={{ color: "#e6edf3" }}>{c.name}</div>
                    <div className="text-[10px] font-mono" style={{ color: "#484f58" }}>{c.id}</div>
                  </div>
                  <div className="w-40">
                    <div className="text-[10px] font-mono mb-0.5" style={{ color: "#484f58" }}>graph score</div>
                    <ScoreBar value={c.graph_score} max={Math.max(...candidates.map(x => x.graph_score), 1)} color="#a855f7" />
                  </div>
                  <Badge variant="purple">{c.graph_score}</Badge>
                </div>
              ))}
            </div>
          )}
          {!loading && !candidates && <Empty message="Enter a user ID and click Query" />}
        </Panel>
      )}
    </div>
  );
}
