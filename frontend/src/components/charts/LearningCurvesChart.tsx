import React, { useState } from 'react';
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
import type { PerModelMetrics } from '../../types/api';

interface LearningCurvesChartProps {
  models: PerModelMetrics[];
}

export const LearningCurvesChart: React.FC<LearningCurvesChartProps> = ({ models }) => {
  const deepModels = models.filter((m) => m.history && m.history.train_loss && m.history.train_loss.length > 0);

  const [selectedModel, setSelectedModel] = useState<string>(
    deepModels[0]?.model || 'cnn1d'
  );

  const currentModel = deepModels.find((m) => m.model === selectedModel) || deepModels[0];

  if (!currentModel || !currentModel.history) {
    return (
      <div className="bg-[#111827] border border-slate-800 rounded-xl p-5 text-center text-slate-400 text-xs">
        No epoch training history available.
      </div>
    );
  }

  const { train_loss, test_loss, test_acc } = currentModel.history;
  const epochCount = train_loss.length;

  const chartData = Array.from({ length: epochCount }, (_, i) => {
    return {
      epoch: `E${i + 1}`,
      epochNum: i + 1,
      trainLoss: Number(train_loss[i]?.toFixed(4)),
      testLoss: Number(test_loss[i]?.toFixed(4)),
      testAcc: Number(((test_acc[i] || 0) * 100).toFixed(2)),
    };
  });

  return (
    <div className="bg-[#111827] border border-slate-800 rounded-xl p-5 shadow-sm space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <h3 className="text-base font-bold text-white tracking-wide">
            Training & Validation Learning Curves
          </h3>
          <p className="text-xs text-slate-400">
            Cross-entropy loss and test accuracy progression over 20 epochs with Cosine Annealing
          </p>
        </div>

        {/* Model Selector Tabs */}
        <div className="flex items-center gap-1.5 bg-[#0b0f17] border border-slate-800 p-1 rounded-lg">
          {deepModels.map((m) => (
            <button
              key={m.model}
              onClick={() => setSelectedModel(m.model)}
              className={`text-xs px-3 py-1 rounded font-medium transition-colors ${
                selectedModel === m.model
                  ? 'bg-blue-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              {m.display_name}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Loss Convergence */}
        <div className="space-y-2">
          <div className="flex justify-between text-xs text-slate-400">
            <span className="font-semibold text-slate-300">Loss Convergence (Cross-Entropy)</span>
            <span>Lower is Better</span>
          </div>
          <div className="h-60 w-full bg-[#0b0f17]/50 border border-slate-800/60 rounded-lg p-2">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 10, right: 15, left: -15, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="epoch" stroke="#64748b" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                <YAxis stroke="#64748b" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#0f172a',
                    border: '1px solid #334155',
                    borderRadius: '8px',
                    color: '#f8fafc',
                    fontSize: '12px',
                  }}
                />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '5px' }} />
                <Line
                  type="monotone"
                  dataKey="trainLoss"
                  name="Train Loss"
                  stroke="#3b82f6"
                  strokeWidth={2}
                  dot={{ r: 2 }}
                />
                <Line
                  type="monotone"
                  dataKey="testLoss"
                  name="Test Loss"
                  stroke="#f43f5e"
                  strokeWidth={2}
                  strokeDasharray="4 4"
                  dot={{ r: 2 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Accuracy Progression */}
        <div className="space-y-2">
          <div className="flex justify-between text-xs text-slate-400">
            <span className="font-semibold text-slate-300">Test Set Accuracy (%)</span>
            <span>Final: {chartData[chartData.length - 1]?.testAcc}%</span>
          </div>
          <div className="h-60 w-full bg-[#0b0f17]/50 border border-slate-800/60 rounded-lg p-2">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 10, right: 15, left: -15, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="epoch" stroke="#64748b" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                <YAxis
                  domain={[80, 96]}
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
                  formatter={(val: any) => [`${val}%`, 'Test Accuracy']}
                />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '5px' }} />
                <Line
                  type="monotone"
                  dataKey="testAcc"
                  name="Test Accuracy"
                  stroke="#10b981"
                  strokeWidth={2}
                  dot={{ r: 2 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
};
