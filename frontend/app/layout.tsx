import type { Metadata } from "next";
import "./globals.css";
import Sidebar from "@/components/Sidebar";
import NetworkBackground from "@/components/NetworkBackground";

export const metadata: Metadata = {
  title: "GraphIQ — Social Graph Intelligence",
  description: "GNN + GraphRAG + Multi-Agent KAG Pipeline",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <NetworkBackground />
        <div className="fixed inset-0 cyber-grid pointer-events-none z-0 opacity-40" />
        <div
          className="fixed inset-0 pointer-events-none z-0"
          style={{ background: "radial-gradient(ellipse at center, transparent 30%, rgba(3,7,18,0.8) 100%)" }}
        />
        <div className="flex min-h-screen relative z-10">
          <Sidebar />
          <main className="ml-[220px] flex-1 min-h-screen p-8 pt-6">
            {children}
          </main>
        </div>
      </body>
    </html>
  );
}
