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
} from 'recharts';
import type { WindowPrediction } from '../../types/api';

interface ActivityDistributionProps {
  windows: WindowPrediction[];
}

const ACTIVITY_COLORS: Record<string, string> = {
  walking: '#3b82f6',
  'walking-upstairs': '#6366f1',
  'walking-downstairs': '#06b6d4',
  sitting: '#f59e0b',
  standing: '#a855f7',
  laying: '#10b981',
};

export const ActivityDistributionPie: React.FC<ActivityDistributionProps> = ({ windows }) => {
  if (!windows || windows.length === 0) return null;

  const counts: Record<string, { count: number; label: string }> = {};

  for (const win of windows) {
    const act = win.consensus_activity;
    if (!counts[act]) {
      counts[act] = { count: 0, label: win.consensus_label };
    }
    counts[act].count += 1;
  }

  const data = Object.entries(counts).map(([act, { count, label }]) => ({
    activity: act,
    name: label,
    count,
    percentage: Number(((count / windows.length) * 100).toFixed(1)),
  }));

  return (
    <div className="bg-[#111827] border border-slate-800 rounded-xl p-5 shadow-sm">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-sm font-bold text-white tracking-wide">
            Predicted Activity Breakdown
          </h3>
          <p className="text-xs text-slate-400">
            Window consensus distribution across {windows.length} segment{windows.length > 1 ? 's' : ''}
          </p>
        </div>
        <span className="text-xs font-mono text-slate-400">
          Total: {windows.length} window{windows.length > 1 ? 's' : ''}
        </span>
      </div>

      <div className="h-48 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ top: 5, right: 20, left: 25, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis type="number" stroke="#64748b" tick={{ fill: '#94a3b8', fontSize: 11 }} unit="%" />
            <YAxis
              type="category"
              dataKey="name"
              stroke="#64748b"
              tick={{ fill: '#94a3b8', fontSize: 11 }}
              width={100}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#0f172a',
                border: '1px solid #334155',
                borderRadius: '8px',
                color: '#f8fafc',
                fontSize: '12px',
              }}
              formatter={(val: any, name: any, item: any) => [
                `${val}% (${item.payload.count} windows)`,
                'Share',
              ]}
            />
            <Bar dataKey="percentage" radius={[0, 4, 4, 0]}>
              {data.map((entry) => (
                <Cell
                  key={`dist-${entry.activity}`}
                  fill={ACTIVITY_COLORS[entry.activity] || '#3b82f6'}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
