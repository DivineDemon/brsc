import React from 'react';
import { Sliders, Cpu, Network, Gauge, RotateCw } from 'lucide-react';
import { ConfigState } from '../types';

interface ControlPanelProps {
  config: ConfigState;
  setConfig: React.Dispatch<React.SetStateAction<ConfigState>>;
  onRunEval: () => void;
  onReindex: () => void;
  isEvaluating: boolean;
}

export const ControlPanel: React.FC<ControlPanelProps> = ({
  config,
  setConfig,
  onRunEval,
  onReindex,
  isEvaluating
}) => {
  return (
    <div className="bg-cardBg backdrop-blur-xl border border-cardBorder rounded-2xl p-6 shadow-xl space-y-6">
      <div className="flex items-center gap-3 border-b border-cardBorder pb-4">
        <Sliders className="w-5 h-5 text-neonPurple" />
        <h2 className="text-lg font-bold text-white">MLOps Control Panel</h2>
      </div>

      {/* Model Switcher */}
      <div className="space-y-2">
        <label className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
          Model Precision Mode
        </label>
        <div className="grid grid-cols-2 gap-2 bg-slate-900/90 p-1 rounded-xl border border-slate-800">
          <button
            onClick={() => setConfig(prev => ({ ...prev, precision: 'int8' }))}
            className={`flex items-center justify-center gap-2 py-2 px-3 rounded-lg text-xs font-semibold transition-all ${
              config.precision === 'int8'
                ? 'bg-gradient-to-r from-neonPurple to-indigo-600 text-white shadow-lg shadow-neonPurple/30'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Cpu className="w-3.5 h-3.5" />
            INT8 (Quantized)
          </button>

          <button
            onClick={() => setConfig(prev => ({ ...prev, precision: 'fp32' }))}
            className={`flex items-center justify-center gap-2 py-2 px-3 rounded-lg text-xs font-semibold transition-all ${
              config.precision === 'fp32'
                ? 'bg-gradient-to-r from-neonPurple to-indigo-600 text-white shadow-lg shadow-neonPurple/30'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Cpu className="w-3.5 h-3.5" />
            FP32 (Standard)
          </button>
        </div>
        <p className="text-[11px] text-slate-500">INT8 cuts memory footprint by ~75% via ONNX Runtime.</p>
      </div>

      {/* Search Engine Switcher */}
      <div className="space-y-2">
        <label className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
          Retrieval Search Engine
        </label>
        <div className="grid grid-cols-2 gap-2 bg-slate-900/90 p-1 rounded-xl border border-slate-800">
          <button
            onClick={() => setConfig(prev => ({ ...prev, engine: 'hnsw' }))}
            className={`flex items-center justify-center gap-2 py-2 px-3 rounded-lg text-xs font-semibold transition-all ${
              config.engine === 'hnsw'
                ? 'bg-gradient-to-r from-neonCyan to-blue-600 text-slate-950 font-bold shadow-lg shadow-neonCyan/20'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Network className="w-3.5 h-3.5" />
            FAISS HNSW
          </button>

          <button
            onClick={() => setConfig(prev => ({ ...prev, engine: 'brute_force' }))}
            className={`flex items-center justify-center gap-2 py-2 px-3 rounded-lg text-xs font-semibold transition-all ${
              config.engine === 'brute_force'
                ? 'bg-gradient-to-r from-neonCyan to-blue-600 text-slate-950 font-bold shadow-lg shadow-neonCyan/20'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Network className="w-3.5 h-3.5" />
            Brute-Force
          </button>
        </div>
        <p className="text-[11px] text-slate-500">HNSW achieves &lt;60ms retrieval vs. 340ms brute-force scans.</p>
      </div>

      {/* High Threshold Slider */}
      <div className="space-y-2">
        <div className="flex justify-between items-center text-xs">
          <label className="font-semibold text-slate-300">High Threshold (&theta;<sub>high</sub>)</label>
          <span className="font-mono font-bold text-neonCyan">{config.thresholdHigh.toFixed(2)}</span>
        </div>
        <input
          type="range"
          min="0.5"
          max="0.99"
          step="0.01"
          value={config.thresholdHigh}
          onChange={e => setConfig(prev => ({ ...prev, thresholdHigh: parseFloat(e.target.value) }))}
          className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-neonCyan"
        />
        <p className="text-[11px] text-slate-500">Above this score serves exact vetted answers directly.</p>
      </div>

      {/* Low Threshold Slider */}
      <div className="space-y-2">
        <div className="flex justify-between items-center text-xs">
          <label className="font-semibold text-slate-300">Low Threshold (&theta;<sub>low</sub>)</label>
          <span className="font-mono font-bold text-neonPurple">{config.thresholdLow.toFixed(2)}</span>
        </div>
        <input
          type="range"
          min="0.2"
          max="0.75"
          step="0.01"
          value={config.thresholdLow}
          onChange={e => setConfig(prev => ({ ...prev, thresholdLow: parseFloat(e.target.value) }))}
          className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-neonPurple"
        />
        <p className="text-[11px] text-slate-500">Below this score escalates query to human agent.</p>
      </div>

      {/* Action Buttons */}
      <div className="space-y-2 pt-2 border-t border-cardBorder">
        <button
          onClick={onRunEval}
          disabled={isEvaluating}
          className="w-full py-2.5 px-4 bg-gradient-to-r from-neonPurple via-indigo-600 to-neonCyan text-white font-bold text-xs rounded-xl shadow-lg shadow-neonPurple/20 hover:opacity-90 active:scale-[0.98] transition-all flex items-center justify-center gap-2 disabled:opacity-50"
        >
          {isEvaluating ? (
            <>
              <RotateCw className="w-4 h-4 animate-spin" />
              Executing Eval Harness...
            </>
          ) : (
            <>
              <Gauge className="w-4 h-4" />
              Run Release Evaluation
            </>
          )}
        </button>

        <button
          onClick={onReindex}
          className="w-full py-2 px-4 bg-slate-900/80 border border-slate-800 hover:border-slate-700 text-slate-300 font-medium text-xs rounded-xl hover:text-white transition-all flex items-center justify-center gap-2"
        >
          <RotateCw className="w-3.5 h-3.5 text-slate-400" />
          Rebuild Vector Index
        </button>
      </div>
    </div>
  );
};
