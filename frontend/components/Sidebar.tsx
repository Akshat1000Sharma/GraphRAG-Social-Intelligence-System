"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Activity,
  MessageSquare,
  GitBranch,
  TrendingUp,
  Users,
  Database,
  Zap,
  Network,
} from "lucide-react";

const navItems = [
  { href: "/", icon: Activity, label: "Dashboard", badge: null },
  { href: "/chat", icon: MessageSquare, label: "Chat Query", badge: "AI" },
  { href: "/explore", icon: Network, label: "Graph Explorer", badge: null },
  { href: "/analytics", icon: TrendingUp, label: "Analytics", badge: null },
  { href: "/users", icon: Users, label: "Users", badge: null },
  { href: "/connections", icon: GitBranch, label: "Connections", badge: null },
  { href: "/admin", icon: Database, label: "Admin", badge: null },
];

export default function Sidebar() {
  const path = usePathname();

  return (
    <aside className="fixed left-0 top-0 h-full w-[220px] z-40 flex flex-col"
      style={{ background: "rgba(13,17,23,0.95)", borderRight: "1px solid rgba(255,255,255,0.06)" }}>

      {/* Logo */}
      <div className="px-5 py-5 border-b" style={{ borderColor: "rgba(255,255,255,0.06)" }}>
        <div className="flex items-center gap-3">
          <div className="relative w-8 h-8">
            <div className="absolute inset-0 rounded-lg"
              style={{ background: "linear-gradient(135deg, #00d4ff22, #a855f722)" }} />
            <Zap className="absolute inset-0 m-auto w-4 h-4" style={{ color: "#00d4ff" }} />
          </div>
          <div>
            <div className="text-sm font-bold font-display" style={{ color: "#e6edf3", letterSpacing: "-0.02em" }}>
              GraphIQ
            </div>
            <div className="text-[10px] font-mono" style={{ color: "#484f58" }}>
              v3.0.0
            </div>
          </div>
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 px-3 py-4 space-y-0.5 overflow-y-auto">
        {navItems.map(({ href, icon: Icon, label, badge }) => {
          const active = path === href;
          return (
            <Link
              key={href}
              href={href}
              className="flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-all duration-150 group relative"
              style={{
                color: active ? "#00d4ff" : "#8b949e",
                background: active ? "rgba(0,212,255,0.08)" : "transparent",
              }}
            >
              {active && (
                <div className="absolute left-0 top-1/2 -translate-y-1/2 w-0.5 h-5 rounded-r"
                  style={{ background: "#00d4ff" }} />
              )}
              <Icon className="w-4 h-4 flex-shrink-0 transition-colors"
                style={{ color: active ? "#00d4ff" : "#484f58" }} />
              <span className="font-body flex-1">{label}</span>
              {badge && (
                <span className="text-[9px] font-mono px-1.5 py-0.5 rounded"
                  style={{ background: "rgba(0,212,255,0.15)", color: "#00d4ff" }}>
                  {badge}
                </span>
              )}
            </Link>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="px-5 py-4 border-t" style={{ borderColor: "rgba(255,255,255,0.06)" }}>
        <div className="text-[10px] font-mono leading-relaxed" style={{ color: "#484f58" }}>
          <div>GNN + GraphRAG</div>
          <div>Multi-Agent KAG Pipeline</div>
        </div>
      </div>
    </aside>
  );
}
