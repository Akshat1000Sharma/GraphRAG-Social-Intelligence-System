"use client";
import { useState, useRef, useEffect } from "react";
import { Send, Bot, User, Clock, Zap } from "lucide-react";
import { api, ChatResponse } from "@/lib/api";
import { GNN_MODEL_OPTIONS } from "@/lib/dataset-options";
import { Panel, Badge, Button, Select, JsonBlock, SectionHeader } from "@/components/ui";

interface Message {
  role: "user" | "assistant";
  content: string;
  data?: ChatResponse;
  ts: number;
}

const DATASET_OPTIONS = [
  { value: "all", label: "All Datasets" },
  { value: "facebook", label: "Facebook (GNN + graph)" },
  { value: "twitter", label: "Twitter (GNN + graph)" },
  { value: "reddit", label: "Reddit (GNN + graph)" },
  { value: "demo", label: "Demo" },
];

const MODE_OPTIONS = [
  { value: "hybrid", label: "Hybrid (Graph + Vector)" },
  { value: "graph", label: "Graph Only" },
  { value: "vector", label: "Vector Only" },
];

const SUGGESTED = [
  "Who are the most influential users in the facebook dataset?",
  "What are the trending posts in the tech topic?",
  "Show me community hubs with high connectivity",
  "Find users with the most mutual connections",
  "Which content creators have the best engagement?",
];

function TimingBar({ ms, max }: { ms: number; max: number }) {
  const pct = Math.min((ms / max) * 100, 100);
  return (
    <div className="h-1 rounded-full overflow-hidden" style={{ background: "rgba(255,255,255,0.06)", width: "80px" }}>
      <div className="h-full rounded-full" style={{ width: `${pct}%`, background: "#00d4ff" }} />
    </div>
  );
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [dataset, setDataset] = useState("all");
  /** Empty = let backend map GNN from `dataset` (all/demo → facebook). */
  const [gnnDataset, setGnnDataset] = useState("");
  const [mode, setMode] = useState("hybrid");
  const [topK, setTopK] = useState("10");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  const send = async () => {
    const msg = input.trim();
    if (!msg || loading) return;
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: msg, ts: Date.now() }]);
    setLoading(true);
    try {
      const res = await api.chat({
        message: msg,
        dataset,
        gnn_dataset: gnnDataset || undefined,
        mode,
        top_k: parseInt(topK),
      });
      setMessages((prev) => [...prev, {
        role: "assistant",
        content: res.insight || "Query processed.",
        data: res,
        ts: Date.now(),
      }]);
    } catch (e) {
      setMessages((prev) => [...prev, { role: "assistant", content: `Error: ${String(e)}`, ts: Date.now() }]);
    } finally {
      setLoading(false);
    }
  };

  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  };

  return (
    <div className="flex flex-col h-screen p-6" style={{ maxWidth: "900px" }}>
      <div className="mb-5 pt-2">
        <div className="flex items-center gap-2 mb-1">
          <span className="text-xs font-mono px-2 py-1 rounded" style={{ background: "rgba(0,212,255,0.1)", color: "#00d4ff" }}>NL QUERY</span>
          <Badge variant="purple">Multi-Agent Pipeline</Badge>
        </div>
        <h1 className="text-2xl font-bold font-display" style={{ color: "#e6edf3", letterSpacing: "-0.03em" }}>Chat Interface</h1>
        <p className="text-xs font-mono mt-0.5" style={{ color: "#484f58" }}>Natural language queries over the social graph</p>
      </div>

      {/* Config bar */}
      <div className="flex items-center gap-3 mb-1 flex-wrap">
        <Select value={dataset} onChange={setDataset} options={DATASET_OPTIONS} />
        <Select value={mode} onChange={setMode} options={MODE_OPTIONS} />
        <Select value={gnnDataset} onChange={setGnnDataset} options={GNN_MODEL_OPTIONS} />
        <Select value={topK} onChange={setTopK} options={[
          { value: "5", label: "Top 5" },
          { value: "10", label: "Top 10" },
          { value: "20", label: "Top 20" },
          { value: "50", label: "Top 50" },
        ]} />
      </div>
      <p className="text-[10px] font-mono mb-4" style={{ color: "#6e7681" }}>
        <strong>Dataset</strong> scopes the graph. <strong className="text-[#a855f7]">GNN</strong> row: choose
        <em> Twitter / Reddit / Facebook weights</em> explicitly, or <em>Auto</em> to follow Dataset (All/Demo → Facebook GNN).
      </p>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto space-y-4 mb-4 pr-1">
        {messages.length === 0 && (
          <div className="py-8">
            <div className="flex items-center justify-center mb-6">
              <div className="w-12 h-12 rounded-full flex items-center justify-center"
                style={{ background: "rgba(0,212,255,0.08)", border: "1px solid rgba(0,212,255,0.2)" }}>
                <Zap className="w-5 h-5" style={{ color: "#00d4ff" }} />
              </div>
            </div>
            <p className="text-center text-sm font-mono mb-6" style={{ color: "#484f58" }}>
              Ask anything about your social graph data
            </p>
            <div className="grid grid-cols-1 gap-2">
              {SUGGESTED.map((s) => (
                <button
                  key={s}
                  onClick={() => setInput(s)}
                  className="text-left px-4 py-2.5 rounded-lg text-xs font-mono transition-all duration-150"
                  style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.06)", color: "#8b949e" }}
                >
                  → {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((m, i) => (
          <div key={i} className={`flex gap-3 ${m.role === "user" ? "flex-row-reverse" : ""}`}>
            <div className="w-7 h-7 rounded-full flex-shrink-0 flex items-center justify-center mt-0.5"
              style={{ background: m.role === "user" ? "rgba(168,85,247,0.2)" : "rgba(0,212,255,0.12)" }}>
              {m.role === "user"
                ? <User className="w-3.5 h-3.5" style={{ color: "#a855f7" }} />
                : <Bot className="w-3.5 h-3.5" style={{ color: "#00d4ff" }} />}
            </div>

            <div className={`flex-1 max-w-2xl ${m.role === "user" ? "flex flex-col items-end" : ""}`}>
              <div
                className="rounded-xl px-4 py-3 text-sm"
                style={{
                  background: m.role === "user" ? "rgba(168,85,247,0.1)" : "rgba(13,17,23,0.9)",
                  border: `1px solid ${m.role === "user" ? "rgba(168,85,247,0.2)" : "rgba(255,255,255,0.07)"}`,
                  color: "#e6edf3",
                  fontFamily: m.role === "assistant" ? "'DM Sans', sans-serif" : undefined,
                }}
              >
                {m.content}
              </div>

              {m.data && (
                <div className="mt-2 space-y-2 w-full">
                  {/* Intent + dataset */}
                  <div className="flex items-center gap-2 flex-wrap">
                    <Badge variant="cyan">{m.data.intent}</Badge>
                    <Badge variant="purple">{m.data.dataset_queried}</Badge>
                    {m.data.gnn_dataset_used && (
                      <Badge variant="amber" title="Pretrained GNN used for this reply">
                        GNN: {m.data.gnn_dataset_used}
                      </Badge>
                    )}
                    <Badge variant="ghost">{m.data.mode}</Badge>
                    {m.data.pipeline_timing_ms?.total && (
                      <span className="text-[10px] font-mono flex items-center gap-1" style={{ color: "#484f58" }}>
                        <Clock className="w-3 h-3" />{m.data.pipeline_timing_ms.total.toFixed(0)}ms
                      </span>
                    )}
                  </div>

                  {/* Results table */}
                  {m.data.results.length > 0 && (
                    <Panel className="p-3">
                      <div className="text-[10px] font-mono mb-2" style={{ color: "#484f58" }}>
                        {m.data.results.length} results
                      </div>
                      <div className="space-y-1.5">
                        {m.data.results.slice(0, 8).map((r, ri) => (
                          <div key={ri} className="flex items-center justify-between px-2 py-1.5 rounded text-xs font-mono"
                            style={{ background: "rgba(255,255,255,0.03)" }}>
                            <span style={{ color: "#e6edf3" }}>{(r.name as string) || (r.id as string) || `#${ri + 1}`}</span>
                            <div className="flex gap-2">
                              {typeof r.influence_score === "number" && (
                                <span style={{ color: "#00d4ff" }}>{(r.influence_score as number).toFixed(2)}</span>
                              )}
                              {typeof r.gnn_score === "number" && (
                                <span style={{ color: "#a855f7" }}>gnn:{(r.gnn_score as number).toFixed(2)}</span>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                    </Panel>
                  )}

                  {/* Timings */}
                  {m.data.pipeline_timing_ms && (
                    <div className="px-1">
                      <div className="text-[10px] font-mono mb-1.5" style={{ color: "#484f58" }}>Pipeline timing</div>
                      <div className="grid grid-cols-3 gap-1.5">
                        {Object.entries(m.data.pipeline_timing_ms).filter(([k]) => k !== "total").map(([k, v]) => (
                          <div key={k} className="flex items-center gap-2">
                            <span className="text-[9px] font-mono w-16" style={{ color: "#484f58" }}>{k}</span>
                            <TimingBar ms={v as number} max={(m.data?.pipeline_timing_ms?.total || 1)} />
                            <span className="text-[9px] font-mono" style={{ color: "#8b949e" }}>{(v as number).toFixed(1)}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  <JsonBlock data={m.data} />
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex gap-3">
            <div className="w-7 h-7 rounded-full flex-shrink-0 flex items-center justify-center"
              style={{ background: "rgba(0,212,255,0.12)" }}>
              <Bot className="w-3.5 h-3.5 animate-pulse" style={{ color: "#00d4ff" }} />
            </div>
            <div className="px-4 py-3 rounded-xl text-xs font-mono"
              style={{ background: "rgba(13,17,23,0.9)", border: "1px solid rgba(255,255,255,0.07)", color: "#484f58" }}>
              <span className="animate-pulse">Running pipeline</span>
              <span className="cursor-blink" />
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="relative">
        <div className="flex gap-3 p-3 rounded-xl" style={{ background: "rgba(13,17,23,0.9)", border: "1px solid rgba(255,255,255,0.08)" }}>
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKey}
            placeholder="Ask about users, connections, trends…"
            rows={2}
            className="flex-1 bg-transparent text-sm font-mono resize-none outline-none"
            style={{ color: "#e6edf3", caretColor: "#00d4ff" }}
          />
          <Button onClick={send} loading={loading} disabled={!input.trim()} size="sm">
            <Send className="w-3.5 h-3.5" />
          </Button>
        </div>
        <div className="text-[10px] font-mono mt-1 text-right" style={{ color: "#484f58" }}>
          Enter to send · Shift+Enter for newline
        </div>
      </div>
    </div>
  );
}
