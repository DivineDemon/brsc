import React from 'react';
import { Table, RotateCw, ThumbsUp, ThumbsDown } from 'lucide-react';
import { RelevanceLog } from '../types';

interface LogsTableProps {
  logs: RelevanceLog[];
  onRefresh: () => void;
}

export const LogsTable: React.FC<LogsTableProps> = ({ logs, onRefresh }) => {
  return (
    <div className="bg-cardBg backdrop-blur-xl border border-cardBorder rounded-2xl p-6 shadow-xl space-y-4">
      <div className="flex items-center justify-between border-b border-cardBorder pb-4">
        <div className="flex items-center gap-3">
          <Table className="w-5 h-5 text-neonCyan" />
          <h2 className="text-lg font-bold text-white">Automated Relevance Logs</h2>
          <span className="px-2.5 py-0.5 bg-slate-900 border border-slate-800 rounded-full text-xs font-mono text-slate-300">
            {logs.length} Queries
          </span>
        </div>

        <button
          onClick={onRefresh}
          className="flex items-center gap-2 px-3 py-1.5 bg-slate-900 border border-slate-800 hover:border-slate-700 rounded-xl text-xs text-slate-300 hover:text-white transition-all"
        >
          <RotateCw className="w-3.5 h-3.5" />
          Refresh Logs
        </button>
      </div>

      <div className="overflow-x-auto max-h-60">
        <table className="w-full text-left border-collapse text-xs">
          <thead>
            <tr className="border-b border-cardBorder text-slate-400 font-semibold uppercase text-[10px]">
              <th className="py-2 px-3">Timestamp</th>
              <th className="py-2 px-3">Query</th>
              <th className="py-2 px-3">Lang</th>
              <th className="py-2 px-3">Matched FAQ</th>
              <th className="py-2 px-3">Score</th>
              <th className="py-2 px-3">Latency</th>
              <th className="py-2 px-3">Method</th>
              <th className="py-2 px-3">Precision</th>
              <th className="py-2 px-3">Status</th>
              <th className="py-2 px-3">Feedback</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-cardBorder/50 font-mono text-slate-300">
            {logs.length === 0 ? (
              <tr>
                <td colSpan={10} className="py-8 text-center text-slate-500 font-sans text-xs">
                  No relevance transaction logs recorded yet. Start a conversation above to trigger automated logging.
                </td>
              </tr>
            ) : (
              logs.map((log, i) => {
                const timeStr = new Date(log.timestamp).toLocaleTimeString([], {
                  hour: '2-digit',
                  minute: '2-digit',
                  second: '2-digit'
                });

                return (
                  <tr key={log.id || i} className="hover:bg-slate-900/50 transition-colors">
                    <td className="py-2.5 px-3 text-slate-500">{timeStr}</td>
                    <td className="py-2.5 px-3 font-sans text-white font-medium max-w-[200px] truncate" title={log.query}>
                      {log.query}
                    </td>
                    <td className="py-2.5 px-3 font-semibold text-slate-300">
                      {log.detected_language === 'ur' ? 'Urdu' : 'English'}
                    </td>
                    <td className="py-2.5 px-3 text-cyan-400">{log.matched_faq_id || 'N/A'}</td>
                    <td className="py-2.5 px-3 text-emerald-400 font-bold">
                      {log.similarity_score ? `${(log.similarity_score * 100).toFixed(0)}%` : 'N/A'}
                    </td>
                    <td className="py-2.5 px-3 text-slate-400">{log.latency_ms.toFixed(1)}ms</td>
                    <td className="py-2.5 px-3 uppercase text-[10px] text-slate-400">{log.search_method}</td>
                    <td className="py-2.5 px-3 uppercase text-[10px] text-slate-400">{log.precision_mode}</td>
                    <td className="py-2.5 px-3">
                      <span
                        className={`px-2 py-0.5 rounded-full text-[10px] font-bold uppercase font-sans ${
                          log.status === 'contained_vetted'
                            ? 'bg-emerald-950/80 text-emerald-400 border border-emerald-500/40'
                            : log.status === 'contained_rag'
                            ? 'bg-cyan-950/80 text-cyan-400 border border-cyan-500/40'
                            : 'bg-rose-950/80 text-rose-400 border border-rose-500/40'
                        }`}
                      >
                        {log.status === 'contained_vetted' ? 'Vetted' : log.status === 'contained_rag' ? 'RAG' : 'Escalated'}
                      </span>
                    </td>
                    <td className="py-2.5 px-3">
                      {log.feedback === 'like' ? (
                        <span className="flex items-center gap-1 text-emerald-400 text-[11px]">
                          <ThumbsUp className="w-3 h-3" /> Like
                        </span>
                      ) : log.feedback === 'dislike' ? (
                        <span className="flex items-center gap-1 text-rose-400 text-[11px]">
                          <ThumbsDown className="w-3 h-3" /> Dislike
                        </span>
                      ) : (
                        <span className="text-slate-600">--</span>
                      )}
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};
