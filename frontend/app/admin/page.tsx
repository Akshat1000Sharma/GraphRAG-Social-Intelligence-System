"use client";
import { useState } from "react";
import { Database, RefreshCw, Plus, User, GitBranch, FileText, Loader2, CheckCircle, XCircle } from "lucide-react";
import { api, InsertResult, NLParseResponse } from "@/lib/api";
import { Panel, Badge, Button, Input, Select, SectionHeader, Tabs, JsonBlock, StatusDot } from "@/components/ui";

function ResultBanner({ result }: { result: InsertResult }) {
  return (
    <div className="p-3 rounded-xl text-xs font-mono space-y-1"
      style={{ background: result.ok ? "rgba(16,185,129,0.08)" : "rgba(239,68,68,0.08)", border: `1px solid ${result.ok ? "rgba(16,185,129,0.2)" : "rgba(239,68,68,0.2)"}` }}>
      <div className="flex items-center gap-2">
        {result.ok ? <CheckCircle className="w-3.5 h-3.5" style={{ color: "#10b981" }} /> : <XCircle className="w-3.5 h-3.5" style={{ color: "#ef4444" }} />}
        <span style={{ color: result.ok ? "#10b981" : "#ef4444" }}>{result.operation} — {result.ok ? "Success" : "Failed"}</span>
      </div>
      {result.nodes_created > 0 && <div style={{ color: "#8b949e" }}>Nodes created: {result.nodes_created}</div>}
      {result.edges_created > 0 && <div style={{ color: "#8b949e" }}>Edges created: {result.edges_created}</div>}
      {result.error && <div style={{ color: "#ef4444" }}>{result.error}</div>}
      {result.cypher_summary && <div style={{ color: "#484f58" }}>{result.cypher_summary}</div>}
    </div>
  );
}

export default function AdminPage() {
  const [tab, setTab] = useState("Datasets");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<InsertResult | null>(null);
  const [parseResult, setParseResult] = useState<NLParseResponse | null>(null);
  const [embedResult, setEmbedResult] = useState<Record<string, unknown> | null>(null);
  const [ingestResult, setIngestResult] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState<string | null>(null);

  // NL Insert
  const [nlCmd, setNlCmd] = useState("");
  const [nlDataset, setNlDataset] = useState("demo");
  const [nlConfirm, setNlConfirm] = useState(false);

  // Structured insert
  const [userName, setUserName] = useState("");
  const [userBio, setUserBio] = useState("");
  const [userDataset, setUserDataset] = useState("demo");
  const [userSourceId, setUserSourceId] = useState("");

  // Edge insert
  const [edgeFrom, setEdgeFrom] = useState("");
  const [edgeTo, setEdgeTo] = useState("");
  const [edgeDataset, setEdgeDataset] = useState("demo");
  const [edgeRel, setEdgeRel] = useState("FRIEND");

  // Post insert
  const [postAuthor, setPostAuthor] = useState("");
  const [postTitle, setPostTitle] = useState("");
  const [postContent, setPostContent] = useState("");
  const [postTopic, setPostTopic] = useState("general");
  const [postDataset, setPostDataset] = useState("demo");

  // Ingest
  const [ingestDataset, setIngestDataset] = useState("");

  const datasetOpts = [
    { value: "demo", label: "Demo" }, { value: "facebook", label: "Facebook" },
    { value: "twitter", label: "Twitter" }, { value: "reddit", label: "Reddit" },
  ];

  const wrap = async (fn: () => Promise<void>) => {
    setLoading(true); setError(null); setResult(null); setParseResult(null);
    try { await fn(); } catch (e) { setError(String(e)); } finally { setLoading(false); }
  };

  const parseNL = () => wrap(async () => {
    const r = await api.insertParse(nlCmd, nlDataset || undefined);
    setParseResult(r);
  });

  const executeNL = () => wrap(async () => {
    const r = await api.insertNL(nlCmd, nlDataset || undefined, true);
    setResult(r);
  });

  const insertUser = (confirm: boolean) => wrap(async () => {
    const r = await api.insertUser({ dataset: userDataset, name: userName, bio: userBio, source_id: userSourceId || undefined }, confirm);
    setResult(r);
  });

  const insertEdge = (confirm: boolean) => wrap(async () => {
    const r = await api.insertEdge({ dataset: edgeDataset, from_user_id: edgeFrom, to_user_id: edgeTo, rel_type: edgeRel, bidirectional: true }, confirm);
    setResult(r);
  });

  const insertPost = (confirm: boolean) => wrap(async () => {
    const r = await api.insertPost({ dataset: postDataset, author_source_id: postAuthor, title: postTitle, content: postContent, topic: postTopic }, confirm);
    setResult(r);
  });

  const refreshEmbeddings = (force: boolean) => wrap(async () => {
    const r = await api.refreshEmbeddings(force);
    setEmbedResult(r as unknown as Record<string, unknown>);
  });

  const ingest = () => wrap(async () => {
    const r = await api.ingestDataset(ingestDataset || undefined);
    setIngestResult(r as unknown as Record<string, unknown>);
  });

  return (
    <div className="p-6 max-w-5xl">
      <div className="mb-6 pt-2">
        <span className="text-xs font-mono px-2 py-1 rounded mb-2 inline-block" style={{ background: "rgba(251,191,36,0.1)", color: "#fbbf24" }}>ADMIN</span>
        <h1 className="text-2xl font-bold font-display" style={{ color: "#e6edf3", letterSpacing: "-0.03em" }}>Administration</h1>
        <p className="text-xs font-mono mt-0.5" style={{ color: "#484f58" }}>Dataset management, graph mutations, embedding refresh</p>
      </div>

      <div className="mb-5">
        <Tabs tabs={["Datasets", "NL Insert", "Insert User", "Insert Edge", "Insert Post", "Embeddings"]} active={tab} onChange={setTab} />
      </div>

      {error && (
        <div className="mb-4 p-3 rounded-lg text-xs font-mono" style={{ background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.2)", color: "#ef4444" }}>
          {error}
        </div>
      )}
      {result && <div className="mb-4"><ResultBanner result={result} /></div>}

      {/* ─── Datasets ─────────────────────────────────────────────────────── */}
      {tab === "Datasets" && (
        <Panel className="p-5">
          <SectionHeader title="Dataset Management" sub="Ingest and manage graph datasets" />
          <div className="flex items-center gap-3 mb-4 flex-wrap">
            <Select value={ingestDataset} onChange={setIngestDataset} options={[
              { value: "", label: "All Datasets" }, ...datasetOpts,
            ]} />
            <Button onClick={ingest} loading={loading} variant="primary">
              <Database className="w-3.5 h-3.5" /> Ingest
            </Button>
          </div>
          {ingestResult && (
            <div className="p-3 rounded-xl" style={{ background: "rgba(16,185,129,0.06)", border: "1px solid rgba(16,185,129,0.15)" }}>
              <div className="text-xs font-mono mb-1" style={{ color: "#10b981" }}>Ingest Results</div>
              {(ingestResult.triggered as string[])?.map((d) => (
                <div key={d} className="flex items-center gap-2 py-1">
                  <StatusDot ok={true} />
                  <span className="text-xs font-mono" style={{ color: "#8b949e" }}>{d}: {String((ingestResult.results as Record<string, unknown>)?.[d] || "ok")}</span>
                </div>
              ))}
              <JsonBlock data={ingestResult} />
            </div>
          )}
          <div className="mt-4 pt-4 border-t" style={{ borderColor: "rgba(255,255,255,0.06)" }}>
            <div className="text-xs font-mono mb-3" style={{ color: "#484f58" }}>⚠ Destructive operations</div>
            <p className="text-xs" style={{ color: "#8b949e" }}>
              Use the ingest button to load or refresh dataset nodes and edges. The API applies a force flag to re-ingest existing data. Check the /datasets/status endpoint via the health dashboard for per-dataset file and count details.
            </p>
          </div>
        </Panel>
      )}

      {/* ─── NL Insert ──────────────────────────────────────────────────────── */}
      {tab === "NL Insert" && (
        <Panel className="p-5">
          <SectionHeader title="Natural Language Insert" sub="Parse and execute graph mutations via NL" />
          <div className="space-y-3 mb-4">
            <div>
              <label className="text-xs font-mono mb-1 block" style={{ color: "#484f58" }}>Command</label>
              <textarea
                value={nlCmd}
                onChange={(e) => setNlCmd(e.target.value)}
                placeholder='e.g. "Add user Alice who is friends with Bob in the facebook dataset"'
                rows={3}
                className="w-full rounded-lg px-3 py-2.5 text-sm font-mono resize-none outline-none"
                style={{ background: "rgba(13,17,23,0.8)", border: "1px solid rgba(255,255,255,0.08)", color: "#e6edf3", caretColor: "#00d4ff" }}
              />
            </div>
            <div className="flex items-center gap-3">
              <Select value={nlDataset} onChange={setNlDataset} options={[{ value: "", label: "Auto-detect" }, ...datasetOpts]} />
              <Button onClick={parseNL} loading={loading} variant="ghost">
                Parse (preview)
              </Button>
              <Button onClick={executeNL} loading={loading} variant="primary">
                <Plus className="w-3.5 h-3.5" /> Execute
              </Button>
            </div>
          </div>

          {parseResult && (
            <div className="p-4 rounded-xl space-y-2" style={{ background: "rgba(0,212,255,0.04)", border: "1px solid rgba(0,212,255,0.12)" }}>
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono" style={{ color: "#00d4ff" }}>PARSED OPERATIONS</span>
                <Badge variant="cyan">{parseResult.dataset}</Badge>
              </div>
              {parseResult.parsed_names.length > 0 && (
                <div className="flex gap-2 flex-wrap">
                  {parseResult.parsed_names.map((n) => <Badge key={n} variant="purple">{n}</Badge>)}
                </div>
              )}
              {parseResult.operations.map((op, i) => (
                <div key={i} className="p-2 rounded text-xs font-mono" style={{ background: "rgba(0,0,0,0.3)", color: "#8b949e" }}>
                  {String((op as Record<string, unknown>).type)}: {JSON.stringify((op as Record<string, unknown>).payload)}
                </div>
              ))}
              {parseResult.error && <div className="text-xs font-mono" style={{ color: "#ef4444" }}>{parseResult.error}</div>}
              <JsonBlock data={parseResult} />
            </div>
          )}
        </Panel>
      )}

      {/* ─── Insert User ────────────────────────────────────────────────────── */}
      {tab === "Insert User" && (
        <Panel className="p-5">
          <SectionHeader title="Insert / Merge User" sub="MERGE by dataset + source_id" />
          <div className="grid grid-cols-2 gap-3 mb-4">
            <div>
              <label className="text-xs font-mono mb-1 block" style={{ color: "#484f58" }}>Dataset</label>
              <Select value={userDataset} onChange={setUserDataset} options={datasetOpts} className="w-full" />
            </div>
            <div>
              <label className="text-xs font-mono mb-1 block" style={{ color: "#484f58" }}>Name *</label>
              <Input value={userName} onChange={setUserName} placeholder="Alice" />
            </div>
            <div>
              <label className="text-xs font-mono mb-1 block" style={{ color: "#484f58" }}>Source ID (optional)</label>
              <Input value={userSourceId} onChange={setUserSourceId} placeholder="alice_01" />
            </div>
            <div>
              <label className="text-xs font-mono mb-1 block" style={{ color: "#484f58" }}>Bio</label>
              <Input value={userBio} onChange={setUserBio} placeholder="Data scientist…" />
            </div>
          </div>
          <div className="flex gap-2">
            <Button onClick={() => insertUser(false)} loading={loading} variant="ghost" disabled={!userName}>
              Preview
            </Button>
            <Button onClick={() => insertUser(true)} loading={loading} variant="primary" disabled={!userName}>
              <User className="w-3.5 h-3.5" /> Insert
            </Button>
          </div>
        </Panel>
      )}

      {/* ─── Insert Edge ────────────────────────────────────────────────────── */}
      {tab === "Insert Edge" && (
        <Panel className="p-5">
          <SectionHeader title="Insert Edge" sub="Create relationship between users" />
          <div className="grid grid-cols-2 gap-3 mb-4">
            <div>
              <label className="text-xs font-mono mb-1 block" style={{ color: "#484f58" }}>Dataset</label>
              <Select value={edgeDataset} onChange={setEdgeDataset} options={datasetOpts} className="w-full" />
            </div>
            <div>
              <label className="text-xs font-mono mb-1 block" style={{ color: "#484f58" }}>Relationship</label>
              <Select value={edgeRel} onChange={setEdgeRel} options={[
                { value: "FRIEND", label: "FRIEND" },
                { value: "FOLLOWS", label: "FOLLOWS" },
                { value: "LIKED", label: "LIKED" },
              ]} className="w-full" />
            </div>
            <div>
              <label className="text-xs font-mono mb-1 block" style={{ color: "#00d4ff" }}>From User (source_id) *</label>
              <Input value={edgeFrom} onChange={setEdgeFrom} placeholder="alice_01" />
            </div>
            <div>
              <label className="text-xs font-mono mb-1 block" style={{ color: "#a855f7" }}>To User (source_id) *</label>
              <Input value={edgeTo} onChange={setEdgeTo} placeholder="bob_02" />
            </div>
          </div>
          <div className="flex gap-2">
            <Button onClick={() => insertEdge(false)} loading={loading} variant="ghost" disabled={!edgeFrom || !edgeTo}>
              Preview
            </Button>
            <Button onClick={() => insertEdge(true)} loading={loading} variant="primary" disabled={!edgeFrom || !edgeTo}>
              <GitBranch className="w-3.5 h-3.5" /> Insert Edge
            </Button>
          </div>
        </Panel>
      )}

      {/* ─── Insert Post ────────────────────────────────────────────────────── */}
      {tab === "Insert Post" && (
        <Panel className="p-5">
          <SectionHeader title="Insert Post" sub="Create post with author relationship" />
          <div className="grid grid-cols-2 gap-3 mb-4">
            <div>
              <label className="text-xs font-mono mb-1 block" style={{ color: "#484f58" }}>Dataset</label>
              <Select value={postDataset} onChange={setPostDataset} options={datasetOpts} className="w-full" />
            </div>
            <div>
              <label className="text-xs font-mono mb-1 block" style={{ color: "#484f58" }}>Author Source ID *</label>
              <Input value={postAuthor} onChange={setPostAuthor} placeholder="alice_01" />
            </div>
            <div className="col-span-2">
              <label className="text-xs font-mono mb-1 block" style={{ color: "#484f58" }}>Title *</label>
              <Input value={postTitle} onChange={setPostTitle} placeholder="Post title…" />
            </div>
            <div>
              <label className="text-xs font-mono mb-1 block" style={{ color: "#484f58" }}>Topic</label>
              <Input value={postTopic} onChange={setPostTopic} placeholder="tech, AI, general…" />
            </div>
            <div className="col-span-2">
              <label className="text-xs font-mono mb-1 block" style={{ color: "#484f58" }}>Content</label>
              <textarea
                value={postContent}
                onChange={(e) => setPostContent(e.target.value)}
                placeholder="Post body text…"
                rows={3}
                className="w-full rounded-lg px-3 py-2.5 text-sm font-mono resize-none outline-none"
                style={{ background: "rgba(13,17,23,0.8)", border: "1px solid rgba(255,255,255,0.08)", color: "#e6edf3" }}
              />
            </div>
          </div>
          <div className="flex gap-2">
            <Button onClick={() => insertPost(false)} loading={loading} variant="ghost" disabled={!postTitle || !postAuthor}>
              Preview
            </Button>
            <Button onClick={() => insertPost(true)} loading={loading} variant="primary" disabled={!postTitle || !postAuthor}>
              <FileText className="w-3.5 h-3.5" /> Insert Post
            </Button>
          </div>
        </Panel>
      )}

      {/* ─── Embeddings ─────────────────────────────────────────────────────── */}
      {tab === "Embeddings" && (
        <Panel className="p-5">
          <SectionHeader title="Embedding Management" sub="Neo4j vector index population" />
          <p className="text-xs mb-4" style={{ color: "#8b949e" }}>
            Refreshes text and GNN embeddings stored on Neo4j nodes. Use <Badge variant="ghost">Incremental</Badge> for fast startup (only new nodes). Use <Badge variant="amber">Force Refresh</Badge> after model changes.
          </p>
          <div className="flex gap-3">
            <Button onClick={() => refreshEmbeddings(false)} loading={loading} variant="primary">
              <RefreshCw className="w-3.5 h-3.5" /> Incremental
            </Button>
            <Button onClick={() => refreshEmbeddings(true)} loading={loading} variant="secondary">
              <RefreshCw className="w-3.5 h-3.5" /> Force Refresh All
            </Button>
          </div>
          {embedResult && (
            <div className="mt-4 p-4 rounded-xl" style={{ background: "rgba(16,185,129,0.06)", border: "1px solid rgba(16,185,129,0.15)" }}>
              <div className="text-xs font-mono mb-2" style={{ color: "#10b981" }}>Refresh Result</div>
              <div className="text-xs font-mono" style={{ color: "#8b949e" }}>
                Status: {String(embedResult.status)}
              </div>
              {embedResult.counts != null && (
                <div className="flex gap-3 mt-1 flex-wrap">
                  {Object.entries(embedResult.counts as Record<string, number>).map(([k, v]) => (
                    <span key={k} className="text-xs font-mono" style={{ color: "#484f58" }}>
                      {k}: <span style={{ color: "#00d4ff" }}>{String(v)}</span>
                    </span>
                  ))}
                </div>
              )}
            </div>
          )}
        </Panel>
      )}
    </div>
  );
}
