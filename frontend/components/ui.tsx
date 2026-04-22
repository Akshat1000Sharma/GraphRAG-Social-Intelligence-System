"use client";
import { ReactNode, useState } from "react";
import { Loader2 } from "lucide-react";

// ─── Panel ────────────────────────────────────────────────────────────────────
export function Panel({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <div
      className={`rounded-xl gradient-border card-hover ${className}`}
      style={{
        background: "rgba(13,17,23,0.9)",
        border: "1px solid rgba(255,255,255,0.07)",
        boxShadow: "0 4px 24px rgba(0,0,0,0.3)",
      }}
    >
      {children}
    </div>
  );
}

// ─── StatCard ─────────────────────────────────────────────────────────────────
export function StatCard({
  label,
  value,
  sub,
  color = "#00d4ff",
  icon,
}: {
  label: string;
  value: string | number;
  sub?: string;
  color?: string;
  icon?: ReactNode;
}) {
  return (
    <div
      className="rounded-xl p-4 card-hover"
      style={{
        background: "rgba(13,17,23,0.9)",
        border: `1px solid ${color}22`,
        boxShadow: `0 4px 24px rgba(0,0,0,0.3), inset 0 0 40px ${color}05`,
      }}
    >
      <div className="flex items-start justify-between mb-2">
        <span className="text-xs font-mono uppercase tracking-wider" style={{ color: "#484f58" }}>
          {label}
        </span>
        {icon && <span style={{ color }}>{icon}</span>}
      </div>
      <div className="text-2xl font-bold font-display" style={{ color, letterSpacing: "-0.03em" }}>
        {value}
      </div>
      {sub && <div className="text-xs mt-1 font-mono" style={{ color: "#484f58" }}>{sub}</div>}
    </div>
  );
}

// ─── Badge ────────────────────────────────────────────────────────────────────
const BADGE_COLORS: Record<string, string> = {
  cyan: "rgba(0,212,255,0.15)",
  purple: "rgba(168,85,247,0.15)",
  green: "rgba(16,185,129,0.15)",
  amber: "rgba(251,191,36,0.15)",
  red: "rgba(239,68,68,0.15)",
};
const BADGE_TEXT: Record<string, string> = {
  cyan: "#00d4ff",
  purple: "#a855f7",
  green: "#10b981",
  amber: "#fbbf24",
  red: "#ef4444",
};

export function Badge({
  children,
  variant = "cyan",
}: {
  children: ReactNode;
  variant?: string;
}) {
  return (
    <span
      className="text-[10px] font-mono px-2 py-0.5 rounded-full"
      style={{ background: BADGE_COLORS[variant] ?? BADGE_COLORS.cyan, color: BADGE_TEXT[variant] ?? BADGE_TEXT.cyan }}
    >
      {children}
    </span>
  );
}

// ─── Button ───────────────────────────────────────────────────────────────────
export function Button({
  children,
  onClick,
  variant = "primary",
  size = "md",
  loading = false,
  disabled = false,
  className = "",
}: {
  children: ReactNode;
  onClick?: () => void;
  variant?: "primary" | "secondary" | "ghost" | "danger";
  size?: "sm" | "md" | "lg";
  loading?: boolean;
  disabled?: boolean;
  className?: string;
}) {
  const styles: Record<string, React.CSSProperties> = {
    primary: {
      background: "rgba(0,212,255,0.12)",
      border: "1px solid rgba(0,212,255,0.3)",
      color: "#00d4ff",
    },
    secondary: {
      background: "rgba(168,85,247,0.12)",
      border: "1px solid rgba(168,85,247,0.3)",
      color: "#a855f7",
    },
    ghost: {
      background: "rgba(255,255,255,0.04)",
      border: "1px solid rgba(255,255,255,0.08)",
      color: "#8b949e",
    },
    danger: {
      background: "rgba(239,68,68,0.12)",
      border: "1px solid rgba(239,68,68,0.3)",
      color: "#ef4444",
    },
  };

  const sizes: Record<string, string> = {
    sm: "px-3 py-1.5 text-xs",
    md: "px-4 py-2 text-sm",
    lg: "px-6 py-3 text-base",
  };

  return (
    <button
      onClick={onClick}
      disabled={disabled || loading}
      className={`rounded-lg font-mono font-medium transition-all duration-150 flex items-center gap-2 ${sizes[size]} ${className}`}
      style={{
        ...styles[variant],
        opacity: disabled || loading ? 0.5 : 1,
        cursor: disabled || loading ? "not-allowed" : "pointer",
      }}
    >
      {loading && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
      {children}
    </button>
  );
}

// ─── Input ────────────────────────────────────────────────────────────────────
export function Input({
  value,
  onChange,
  placeholder,
  className = "",
  multiline = false,
  rows = 3,
}: {
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  className?: string;
  multiline?: boolean;
  rows?: number;
}) {
  const baseStyle: React.CSSProperties = {
    background: "rgba(13,17,23,0.8)",
    border: "1px solid rgba(255,255,255,0.08)",
    color: "#e6edf3",
    outline: "none",
    resize: "none",
  };

  const cls = `w-full rounded-lg px-3 py-2.5 text-sm font-mono placeholder-gray-600 transition-all duration-150 focus:border-cyan-500 ${className}`;

  if (multiline) {
    return (
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        rows={rows}
        className={cls}
        style={baseStyle}
      />
    );
  }
  return (
    <input
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      className={cls}
      style={baseStyle}
    />
  );
}

// ─── Select ───────────────────────────────────────────────────────────────────
export function Select({
  value,
  onChange,
  options,
  className = "",
}: {
  value: string;
  onChange: (v: string) => void;
  options: { value: string; label: string }[];
  className?: string;
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className={`rounded-lg px-3 py-2 text-sm font-mono ${className}`}
      style={{
        background: "rgba(13,17,23,0.8)",
        border: "1px solid rgba(255,255,255,0.08)",
        color: "#e6edf3",
        outline: "none",
      }}
    >
      {options.map((o) => (
        <option key={o.value} value={o.value} style={{ background: "#0d1117" }}>
          {o.label}
        </option>
      ))}
    </select>
  );
}

// ─── SectionHeader ────────────────────────────────────────────────────────────
export function SectionHeader({
  title,
  sub,
  action,
}: {
  title: string;
  sub?: string;
  action?: ReactNode;
}) {
  return (
    <div className="flex items-start justify-between mb-5">
      <div>
        <h2 className="text-lg font-bold font-display" style={{ color: "#e6edf3", letterSpacing: "-0.02em" }}>
          {title}
        </h2>
        {sub && <p className="text-xs mt-0.5 font-mono" style={{ color: "#484f58" }}>{sub}</p>}
      </div>
      {action}
    </div>
  );
}

// ─── Loading skeleton ─────────────────────────────────────────────────────────
export function Skeleton({ className = "" }: { className?: string }) {
  return (
    <div
      className={`rounded shimmer ${className}`}
      style={{ background: "rgba(255,255,255,0.04)", minHeight: 16 }}
    />
  );
}

// ─── Empty state ──────────────────────────────────────────────────────────────
export function Empty({ message }: { message: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-12 text-center">
      <div className="w-10 h-10 rounded-full mb-3 flex items-center justify-center"
        style={{ background: "rgba(255,255,255,0.04)" }}>
        <span className="text-xl">∅</span>
      </div>
      <p className="text-sm font-mono" style={{ color: "#484f58" }}>{message}</p>
    </div>
  );
}

// ─── Tabs ─────────────────────────────────────────────────────────────────────
export function Tabs({
  tabs,
  active,
  onChange,
}: {
  tabs: string[];
  active: string;
  onChange: (t: string) => void;
}) {
  return (
    <div className="flex gap-1 p-1 rounded-lg" style={{ background: "rgba(255,255,255,0.03)" }}>
      {tabs.map((t) => (
        <button
          key={t}
          onClick={() => onChange(t)}
          className="px-3 py-1.5 text-xs font-mono rounded-md transition-all duration-150"
          style={{
            background: active === t ? "rgba(0,212,255,0.12)" : "transparent",
            color: active === t ? "#00d4ff" : "#484f58",
            border: active === t ? "1px solid rgba(0,212,255,0.2)" : "1px solid transparent",
          }}
        >
          {t}
        </button>
      ))}
    </div>
  );
}

// ─── JsonBlock ────────────────────────────────────────────────────────────────
export function JsonBlock({ data }: { data: unknown }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="mt-2">
      <button
        onClick={() => setOpen(!open)}
        className="text-xs font-mono"
        style={{ color: "#484f58" }}
      >
        {open ? "▼" : "▶"} raw json
      </button>
      {open && (
        <pre
          className="mt-2 p-3 rounded-lg text-[11px] font-mono overflow-auto max-h-64"
          style={{ background: "rgba(0,0,0,0.4)", color: "#8b949e", border: "1px solid rgba(255,255,255,0.06)" }}
        >
          {JSON.stringify(data, null, 2)}
        </pre>
      )}
    </div>
  );
}

// ─── StatusDot ────────────────────────────────────────────────────────────────
export function StatusDot({ ok, loading }: { ok: boolean; loading?: boolean }) {
  return (
    <span
      className="status-dot inline-block"
      style={{
        background: loading ? "#fbbf24" : ok ? "#10b981" : "#ef4444",
        boxShadow: loading
          ? "0 0 6px rgba(251,191,36,0.8)"
          : ok
          ? "0 0 6px rgba(16,185,129,0.8)"
          : "0 0 6px rgba(239,68,68,0.8)",
      }}
    />
  );
}
