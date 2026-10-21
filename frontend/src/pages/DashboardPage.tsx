import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Activity,
  Cpu,
  Layers,
  ArrowRight,
  RefreshCw,
  CheckCircle2,
  AlertCircle,
  Zap,
  ShieldCheck,
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { apiService } from '../api/client';
import type { HealthResponse, ModelMetadata } from '../types/api';
import { MetricCard } from '../components/MetricCard';
import { ActivityBadge } from '../components/ActivityBadge';

export const DashboardPage: React.FC = () => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [models, setModels] = useState<ModelMetadata[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadDashboardData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [healthData, modelsData] = await Promise.all([
        apiService.getHealth(),
        apiService.getModels(),
      ]);
      setHealth(healthData);
      setModels(modelsData);
    } catch (err: any) {
      setError(err?.detail || 'Failed to fetch system data from backend.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadDashboardData();
  }, []);

  const convlstmModel = models.find((m) => m.id === 'convlstm');
  const cnnModel = models.find((m) => m.id === 'cnn1d');
  const gruModel = models.find((m) => m.id === 'gru');
  const lstmModel = models.find((m) => m.id === 'lstm');
  const rfModel = models.find((m) => m.id === 'randomforest');

  const chartData = models
    .filter((m) => m.test_accuracy !== null && m.macro_f1 !== null)
    .map((m) => ({
      name: m.name,
      accuracy: Number(((m.test_accuracy || 0) * 100).toFixed(2)),
      macroF1: Number(((m.macro_f1 || 0) * 100).toFixed(2)),
    }));

  const activities = [
    { key: 'walking', label: 'Walking', desc: 'Rhythmic, periodic forward locomotion' },
    { key: 'walking-upstairs', label: 'Walking Upstairs', desc: 'Step ascent with elevated vertical acceleration' },
    { key: 'walking-downstairs', label: 'Walking Downstairs', desc: 'Step descent with sharp downward impact spikes' },
    { key: 'sitting', label: 'Sitting', desc: 'Stationary seated posture with near-zero dynamic linear acceleration' },
    { key: 'standing', label: 'Standing', desc: 'Upright stationary posture with vertical gravity alignment' },
    { key: 'laying', label: 'Laying', desc: 'Horizontal static orientation with gravity vector reoriented by 90°' },
  ];

  return (
    <div className="space-y-6 max-w-6xl mx-auto">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <h1 className="text-2xl font-bold tracking-tight text-white">MotionSense AI</h1>
            <span className="text-xs px-2.5 py-0.5 rounded-full font-mono bg-blue-950/80 text-blue-300 border border-blue-800/80">
              Deep Learning Mini-Project
            </span>
          </div>
          <p className="text-sm text-slate-400">
            Smartphone inertial sensor activity classification using 1D-CNN, GRU, and LSTM against a classical baseline.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={loadDashboardData}
            disabled={loading}
            className="flex items-center gap-1.5 text-xs font-medium text-slate-300 bg-[#111827] border border-slate-800 hover:border-slate-700 px-3 py-2 rounded-lg transition-colors"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin text-blue-400' : ''}`} />
            <span>Refresh</span>
          </button>
          <Link
            to="/analyzer"
            className="flex items-center gap-1.5 text-xs font-semibold text-white bg-blue-600 hover:bg-blue-500 px-3.5 py-2 rounded-lg transition-colors shadow-sm"
          >
            <span>Open Analyzer</span>
            <ArrowRight className="w-4 h-4" />
          </Link>
        </div>
      </div>

      {/* Error state */}
      {error && (
        <div className="bg-rose-950/60 border border-rose-800/80 rounded-xl p-4 flex items-start gap-3 text-rose-300 text-sm">
          <AlertCircle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
          <div>
            <div className="font-semibold text-rose-200">Backend Connection Error</div>
            <p className="mt-0.5 text-xs text-rose-300/90">{error}</p>
          </div>
        </div>
      )}

      {/* System Status Banner */}
      <div className="bg-[#111827] border border-slate-800 rounded-xl p-4 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-slate-800 flex items-center justify-center text-slate-300">
            <Cpu className="w-5 h-5 text-blue-400" />
          </div>
          <div>
            <div className="text-xs text-slate-400 font-medium">Inference Execution Engine</div>
            <div className="text-sm font-bold text-white flex items-center gap-2">
              <span className="capitalize">{health?.device || 'CPU'} Runtime</span>
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-950/80 text-emerald-400 border border-emerald-800/60 font-mono">
                Active
              </span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-6 text-xs text-slate-400">
          <div>
            <span className="block text-slate-500 text-[11px] uppercase font-semibold">
              Deep Models Loaded
            </span>
            <span className="text-slate-200 font-bold text-sm">
              {health?.models_loaded_count ?? 0} / {health?.total_models ?? 3} Ready
            </span>
          </div>

          <div className="hidden sm:block border-l border-slate-800 h-8" />

          <div>
            <span className="block text-slate-500 text-[11px] uppercase font-semibold">
              Backend Status
            </span>
            <span
              className={`font-bold text-sm flex items-center gap-1.5 ${
                health?.status === 'healthy' ? 'text-emerald-400' : 'text-amber-400'
              }`}
            >
              <CheckCircle2 className="w-3.5 h-3.5" />
              <span className="capitalize">{health?.status || 'Unknown'}</span>
            </span>
          </div>

          <div className="hidden sm:block border-l border-slate-800 h-8" />

          <div>
            <span className="block text-slate-500 text-[11px] uppercase font-semibold">
              Benchmark Dataset
            </span>
            <span className="text-slate-200 font-bold text-sm">UCI HAR (2,947 Test Windows)</span>
          </div>
        </div>
      </div>

      {/* Model Benchmark Performance Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3.5">
        <MetricCard
          title="DeepConvLSTM (Advanced)"
          value={convlstmModel ? `${((convlstmModel.test_accuracy || 0) * 100).toFixed(2)}%` : '---'}
          subtitle={convlstmModel ? `Macro-F1: ${((convlstmModel.macro_f1 || 0) * 100).toFixed(2)}%` : 'Loading...'}
          badge={{ text: 'Conv+LSTM+Attn', variant: 'success' }}
          icon={<Zap className="w-4 h-4 text-cyan-400" />}
          loading={loading}
        />

        <MetricCard
          title="1D-CNN (Convolutional)"
          value={cnnModel ? `${((cnnModel.test_accuracy || 0) * 100).toFixed(2)}%` : '---'}
          subtitle={cnnModel ? `Macro-F1: ${((cnnModel.macro_f1 || 0) * 100).toFixed(2)}%` : 'Loading...'}
          badge={{ text: 'Top Baseline', variant: 'info' }}
          icon={<Zap className="w-4 h-4 text-blue-400" />}
          loading={loading}
        />

        <MetricCard
          title="RandomForest (Baseline)"
          value={rfModel ? `${((rfModel.test_accuracy || 0) * 100).toFixed(2)}%` : '---'}
          subtitle={rfModel ? `Macro-F1: ${((rfModel.macro_f1 || 0) * 100).toFixed(2)}%` : 'Loading...'}
          badge={{ text: '561 Features', variant: 'default' }}
          icon={<Layers className="w-4 h-4 text-amber-400" />}
          loading={loading}
        />

        <MetricCard
          title="LSTM (Bidirectional)"
          value={lstmModel ? `${((lstmModel.test_accuracy || 0) * 100).toFixed(2)}%` : '---'}
          subtitle={lstmModel ? `Macro-F1: ${((lstmModel.macro_f1 || 0) * 100).toFixed(2)}%` : 'Loading...'}
          badge={{ text: 'Recurrent', variant: 'default' }}
          icon={<Activity className="w-4 h-4 text-purple-400" />}
          loading={loading}
        />

        <MetricCard
          title="GRU (Bidirectional)"
          value={gruModel ? `${((gruModel.test_accuracy || 0) * 100).toFixed(2)}%` : '---'}
          subtitle={gruModel ? `Macro-F1: ${((gruModel.macro_f1 || 0) * 100).toFixed(2)}%` : 'Loading...'}
          badge={{ text: 'Recurrent', variant: 'default' }}
          icon={<Activity className="w-4 h-4 text-emerald-400" />}
          loading={loading}
        />
      </div>

      {/* Single Clear Model Comparison Chart */}
      <div className="bg-[#111827] border border-slate-800 rounded-xl p-5 shadow-sm">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-4">
          <div>
            <h3 className="text-sm font-bold text-white tracking-wide">
              Architecture Benchmark Comparison
            </h3>
            <p className="text-xs text-slate-400">
              Test Accuracy and Macro-F1 evaluated across 2,947 subject-independent test windows
            </p>
          </div>
          <span className="text-[11px] px-2 py-0.5 rounded font-mono bg-blue-950/80 text-blue-300 border border-blue-800/80 self-start sm:self-auto">
            Higher is Better
          </span>
        </div>

        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 10, right: 10, left: -15, bottom: 5 }}>
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

      {/* Six Supported Activity Classes */}
      <div className="bg-[#111827] border border-slate-800 rounded-xl p-5 shadow-sm">
        <h3 className="text-sm font-bold text-white tracking-wide mb-1">
          Six Target Activity Classes
        </h3>
        <p className="text-xs text-slate-400 mb-4">
          Mutually exclusive human motions classified from tri-axial accelerometer and gyroscope dynamics at 50 Hz
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {activities.map((act) => (
            <div
              key={act.key}
              className="bg-[#0b0f17] border border-slate-800/80 rounded-lg p-3 flex items-start gap-2.5"
            >
              <div className="mt-0.5">
                <ActivityBadge activity={act.key} label={act.label} />
              </div>
              <p className="text-xs text-slate-400 leading-snug">{act.desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Mini-Project Academic Disclaimer */}
      <div className="bg-[#0b0f17] border border-slate-800/80 rounded-xl p-4 flex items-start gap-3 text-xs text-slate-400">
        <ShieldCheck className="w-5 h-5 text-blue-400 shrink-0 mt-0.5" />
        <div>
          <span className="font-semibold text-slate-300">College Mini-Project Scope: </span>
          MotionSense AI is an academic deep-learning exploration comparing convolutional and recurrent representations against classical feature baselines. It is designed for educational demonstration and is not certified for clinical or safety-critical use.
        </div>
      </div>
    </div>
  );
};
