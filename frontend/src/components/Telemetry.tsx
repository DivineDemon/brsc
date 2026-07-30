import React from 'react';
import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
  Legend,
  CategoryScale,
  LinearScale,
  BarElement,
  Title
} from 'chart.js';
import { Doughnut, Bar } from 'react-chartjs-2';
import { SystemStats } from '../types';

ChartJS.register(
  ArcElement,
  Tooltip,
  Legend,
  CategoryScale,
  LinearScale,
  BarElement,
  Title
);

interface TelemetryProps {
  stats: SystemStats;
}

export const Telemetry: React.FC<TelemetryProps> = ({ stats }) => {
  // Doughnut Data (Language Distribution)
  const doughnutData = {
    labels: ['English', 'Urdu'],
    datasets: [
      {
        data: [stats.english_count, stats.urdu_count],
        backgroundColor: ['#8a2be2', '#00f2fe'],
        borderColor: 'rgba(255, 255, 255, 0.05)',
        borderWidth: 1
      }
    ]
  };

  const doughnutOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'right' as const,
        labels: { color: '#94a3b8', font: { family: 'Outfit', size: 11 } }
      },
      title: {
        display: true,
        text: 'Language Query Distribution',
        color: '#fff',
        font: { family: 'Outfit', size: 12, weight: 'bold' as const }
      }
    },
    cutout: '65%'
  };

  // Bar Data (Latency Comparison)
  const barData = {
    labels: ['HNSW', 'Cosine', 'INT8', 'FP32'],
    datasets: [
      {
        label: 'Avg Latency (ms)',
        data: [
          stats.hnsw_avg_latency_ms,
          stats.brute_avg_latency_ms,
          stats.int8_avg_latency_ms,
          stats.fp32_avg_latency_ms
        ],
        backgroundColor: [
          'rgba(0, 242, 254, 0.7)',
          'rgba(100, 116, 139, 0.4)',
          'rgba(138, 43, 226, 0.7)',
          'rgba(100, 116, 139, 0.4)'
        ],
        borderColor: ['#00f2fe', '#475569', '#8a2be2', '#475569'],
        borderWidth: 1
      }
    ]
  };

  const barOptions = {
    indexAxis: 'y' as const,
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      title: {
        display: true,
        text: 'Engine Latency Comparison (ms)',
        color: '#fff',
        font: { family: 'Outfit', size: 12, weight: 'bold' as const }
      }
    },
    scales: {
      x: {
        grid: { color: 'rgba(255, 255, 255, 0.05)' },
        ticks: { color: '#64748b', font: { family: 'Fira Code', size: 10 } }
      },
      y: {
        grid: { display: false },
        ticks: { color: '#94a3b8', font: { family: 'Outfit', size: 11 } }
      }
    }
  };

  return (
    <div className="space-y-6">
      {/* Metric Stat Cards */}
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-cardBg backdrop-blur-xl border border-cardBorder rounded-2xl p-4 shadow-xl">
          <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">Containment Rate</span>
          <div className="text-2xl font-extrabold text-emerald-400 mt-1">
            {stats.containment_rate.toFixed(1)}%
          </div>
          <span className="text-[10px] text-slate-500">Target: &gt;70% (vetted + RAG)</span>
        </div>

        <div className="bg-cardBg backdrop-blur-xl border border-cardBorder rounded-2xl p-4 shadow-xl">
          <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">Avg Latency</span>
          <div className="text-2xl font-extrabold text-neonCyan mt-1">
            {stats.avg_latency_ms.toFixed(1)} ms
          </div>
          <span className="text-[10px] text-slate-500">Target: &lt;180ms</span>
        </div>
      </div>

      {/* Telemetry Charts */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 h-48">
        <div className="bg-cardBg backdrop-blur-xl border border-cardBorder rounded-2xl p-3 shadow-xl flex items-center justify-center">
          <Doughnut data={doughnutData} options={doughnutOptions} />
        </div>

        <div className="bg-cardBg backdrop-blur-xl border border-cardBorder rounded-2xl p-3 shadow-xl flex items-center justify-center">
          <Bar data={barData} options={barOptions} />
        </div>
      </div>
    </div>
  );
};
