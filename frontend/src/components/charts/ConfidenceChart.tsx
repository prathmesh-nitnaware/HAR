import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import type { WindowPrediction } from '../../types/api';

interface ConfidenceChartProps {
  windows: WindowPrediction[];
}

export const ConfidenceChart: React.FC<ConfidenceChartProps> = ({ windows }) => {
  if (!windows || windows.length === 0) return null;

  const data = windows.map((w) => {
    return {
      windowId: `#${w.window_id}`,
      time: `${w.start_time_sec.toFixed(1)}s`,
      consensus: w.consensus_label,
      ConvLSTM: w.models.convlstm ? Number((w.models.convlstm.confidence * 100).toFixed(1)) : null,
      CNN1D: w.models.cnn1d ? Number((w.models.cnn1d.confidence * 100).toFixed(1)) : null,
      GRU: w.models.gru ? Number((w.models.gru.confidence * 100).toFixed(1)) : null,
      LSTM: w.models.lstm ? Number((w.models.lstm.confidence * 100).toFixed(1)) : null,
      unanimous: w.unanimous_agreement,
    };
  });

  return (
    <div className="bg-[#111827] border border-slate-800 rounded-xl p-5 shadow-sm">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-1 mb-4">
        <div>
          <h3 className="text-sm font-bold text-white tracking-wide">
            Model-Reported Prediction Confidence Across Windows
          </h3>
          <p className="text-xs text-slate-400">
            Softmax output probabilities per architecture (Note: model-reported confidence, not correctness guarantee)
          </p>
        </div>
        <span className="text-[11px] px-2 py-0.5 rounded font-mono bg-blue-950/80 text-blue-300 border border-blue-800/80">
          Scale: 0 – 100%
        </span>
      </div>

      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 10, right: 15, left: -15, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis dataKey="windowId" stroke="#64748b" tick={{ fill: '#94a3b8', fontSize: 11 }} />
            <YAxis
              domain={[50, 100]}
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
            <Line
              type="monotone"
              dataKey="ConvLSTM"
              name="DeepConvLSTM Confidence"
              stroke="#06b6d4"
              strokeWidth={2}
              dot={{ r: 3 }}
            />
            <Line
              type="monotone"
              dataKey="CNN1D"
              name="1D-CNN Confidence"
              stroke="#3b82f6"
              strokeWidth={2}
              dot={{ r: 3 }}
            />
            <Line
              type="monotone"
              dataKey="GRU"
              name="GRU Confidence"
              stroke="#10b981"
              strokeWidth={2}
              dot={{ r: 3 }}
            />
            <Line
              type="monotone"
              dataKey="LSTM"
              name="LSTM Confidence"
              stroke="#8b5cf6"
              strokeWidth={2}
              dot={{ r: 3 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
