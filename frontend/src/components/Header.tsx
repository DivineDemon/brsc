import React from 'react';
import { Bot, Cpu, Activity } from 'lucide-react';
import { ConfigState } from '../types';

interface HeaderProps {
  config: ConfigState;
}

export const Header: React.FC<HeaderProps> = ({ config }) => {
  return (
    <header className="flex flex-col md:flex-row justify-between items-start md:items-center p-6 bg-cardBg backdrop-blur-xl border border-cardBorder rounded-2xl gap-4 shadow-2xl">
      <div className="flex items-center gap-4">
        <div className="p-3 bg-gradient-to-br from-neonPurple to-neonCyan rounded-2xl shadow-lg shadow-neonPurple/20">
          <Bot className="w-8 h-8 text-white" />
        </div>
        <div>
          <h1 className="text-3xl font-extrabold tracking-tight bg-gradient-to-r from-white via-slate-200 to-neonCyan bg-clip-text text-transparent">
            BiliRAG
          </h1>
          <p className="text-sm font-medium text-slate-400">
            Bilingual (Urdu/English) RAG Support & MLOps Control Dashboard
          </p>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <div className="flex items-center gap-2 px-3.5 py-1.5 bg-slate-900/80 border border-slate-800 rounded-full text-xs font-mono text-slate-300">
          <Cpu className="w-3.5 h-3.5 text-neonCyan" />
          <span>{config.isStaticSpace ? 'Engine: WebAssembly WASM' : `Engine: ${config.precision.toUpperCase()} ONNX`}</span>
        </div>

        <div className="flex items-center gap-2 px-3.5 py-1.5 bg-emerald-950/60 border border-emerald-500/30 rounded-full text-xs font-semibold text-emerald-400">
          <Activity className="w-3.5 h-3.5 animate-pulse text-emerald-400" />
          <span>System Operational</span>
        </div>
      </div>
    </header>
  );
};
