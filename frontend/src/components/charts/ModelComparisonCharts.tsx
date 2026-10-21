import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  Legend,
} from 'recharts';
import type { ModelMetadata } from '../../types/api';

interface ModelComparisonChartsProps {
  models: ModelMetadata[];
}

const MODEL_COLORS: Record<string, string> = {
  convlstm: '#06b6d4', // Cyan (Advanced)
  cnn1d: '#3b82f6', // Electric blue
  gru: '#10b981',   // Emerald
  lstm: '#8b5cf6',  // Purple
  randomforest: '#f59e0b', // Amber
};

export const ModelComparisonCharts: React.FC<ModelComparisonChartsProps> = ({ models }) => {
  const metricData = models
    .filter((m) => m.test_accuracy !== null && m.macro_f1 !== null)
    .map((m) => ({
      name: m.name,
      id: m.id,
      accuracy: Number(((m.test_accuracy || 0) * 100).toFixed(2)),
      macroF1: Number(((m.macro_f1 || 0) * 100).toFixed(2)),
      isBaseline: m.id === 'randomforest',
    }));

  const paramData = models
    .filter((m) => m.parameters !== null)
    .map((m) => ({
      name: m.name,
      id: m.id,
      parameters: m.parameters || 0,
      parametersK: Number(((m.parameters || 0) / 1000).toFixed(1)),
      diskMb: m.disk_size_mb || 0,
    }));

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Benchmark Accuracy & F1 */}
      <div className="bg-[#111827] border border-slate-800 rounded-xl p-5 shadow-sm">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-sm font-bold text-white tracking-wide">
              Benchmark Accuracy & Macro-F1
            </h3>
            <p className="text-xs text-slate-400">
              Evaluated on 2,947 UCI HAR subject-independent test windows
            </p>
          </div>
          <span className="text-[11px] px-2 py-0.5 rounded font-mono bg-blue-950/80 text-blue-300 border border-blue-800/80">
            Higher is Better
          </span>
        </div>

        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={metricData} margin={{ top: 10, right: 10, left: -15, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="name" stroke="#64748b" tick={{ fill: '#94a3b8', fontSize: 11 }} />
              <YAxis
                domain={[85, 96]}
                stroke="#64748b"
                tick={{ fill: '#94a3b8', fontSize: 11 }}
                unit="%"
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#0f172a',
                  border: '1px solid #334155',
                  borderRadius: '8px',
                  color: '#f8fafc',
                  fontSize: '12px',
                }}
                formatter={(val: any) => [`${val}%`, '']}
              />
              <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
              <Bar dataKey="accuracy" name="Test Accuracy (%)" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              <Bar dataKey="macroF1" name="Macro-F1 (%)" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Parameter Count & Disk Footprint */}
      <div className="bg-[#111827] border border-slate-800 rounded-xl p-5 shadow-sm">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-sm font-bold text-white tracking-wide">
              Deep Architecture Complexity
            </h3>
            <p className="text-xs text-slate-400">
              Trainable parameters & serialized PyTorch checkpoint size
            </p>
          </div>
          <span className="text-[11px] px-2 py-0.5 rounded font-mono bg-slate-800 text-slate-300 border border-slate-700">
            RF Baseline: 561 Features (No Params)
          </span>
        </div>

        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={paramData} margin={{ top: 10, right: 10, left: 10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="name" stroke="#64748b" tick={{ fill: '#94a3b8', fontSize: 11 }} />
              <YAxis stroke="#64748b" tick={{ fill: '#94a3b8', fontSize: 11 }} unit="k" />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#0f172a',
                  border: '1px solid #334155',
                  borderRadius: '8px',
                  color: '#f8fafc',
                  fontSize: '12px',
                }}
                formatter={(val: any, name: any, item: any) => {
                  if (name === 'Trainable Parameters (k)') {
                    return [`${val}k (${item.payload.parameters.toLocaleString()} params)`, name];
                  }
                  return [`${item.payload.diskMb} MB`, 'Checkpoint Size'];
                }}
              />
              <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
              <Bar
                dataKey="parametersK"
                name="Trainable Parameters (k)"
                fill="#10b981"
                radius={[4, 4, 0, 0]}
              >
                {paramData.map((entry) => (
                  <Cell
                    key={`cell-${entry.id}`}
                    fill={MODEL_COLORS[entry.id] || '#10b981'}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};
