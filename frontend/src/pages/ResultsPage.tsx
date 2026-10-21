import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import {
  Download,
  ArrowLeft,
  CheckCircle2,
  AlertTriangle,
  Clock,
  FileSpreadsheet,
  ChevronLeft,
  ChevronRight,
  ShieldCheck,
  RefreshCw,
  Table,
} from 'lucide-react';
import { useAnalysis } from '../context/AnalysisContext';
import { apiService } from '../api/client';
import type { ConfusionMatrixResponse, ModelMetadata } from '../types/api';
import { ActivityBadge } from '../components/ActivityBadge';
import { TimelineChart } from '../components/charts/TimelineChart';
import { ConfidenceChart } from '../components/charts/ConfidenceChart';
import { ConfusionMatrixHeatmap } from '../components/charts/ConfusionMatrixHeatmap';
import { exportPredictionsToCsv, downloadCsvFile } from '../utils/csvExport';

export const ResultsPage: React.FC = () => {
  const { currentAnalysis } = useAnalysis();
  const [currentPage, setCurrentPage] = useState(1);
  const pageSize = 8;

  // Evaluation data for benchmark comparison
  const [cmData, setCmData] = useState<ConfusionMatrixResponse | null>(null);
  const [models, setModels] = useState<ModelMetadata[]>([]);
  const [activeCmModel, setActiveCmModel] = useState<string>('cnn1d');
  const [evalLoading, setEvalLoading] = useState(true);

  useEffect(() => {
    const fetchEvalData = async () => {
      setEvalLoading(true);
      try {
        const [cmRes, modelsRes] = await Promise.all([
          apiService.getConfusionMatrices(),
          apiService.getModels(),
        ]);
        setCmData(cmRes);
        setModels(modelsRes);
      } catch {
        // Soft fail if evaluation endpoints unavailable
      } finally {
        setEvalLoading(false);
      }
    };
    fetchEvalData();
  }, []);

  const handleExportCsv = () => {
    if (!currentAnalysis) return;
    const csvData = exportPredictionsToCsv(currentAnalysis);
    const cleanName = currentAnalysis.source_name.replace(/\.[^/.]+$/, '');
    downloadCsvFile(csvData, `${cleanName}_predictions.csv`);
  };

  const selectedMatrixData =
    cmData?.matrices[activeCmModel] ||
    (cmData ? Object.values(cmData.matrices)[0] : undefined);

  return (
    <div className="space-y-8 max-w-6xl mx-auto">
      {/* ── Section A: Active Prediction Results ── */}
      {currentAnalysis ? (
        <div className="space-y-6">
          {/* Header */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800">
            <div>
              <div className="flex items-center gap-2 mb-1">
                <Link
                  to="/analyzer"
                  className="text-xs text-slate-400 hover:text-white flex items-center gap-1 transition-colors"
                >
                  <ArrowLeft className="w-3.5 h-3.5" />
                  <span>Back to Analyzer</span>
                </Link>
                <span className="text-slate-600">/</span>
                <span className="text-xs font-mono text-blue-400">
                  {currentAnalysis.n_windows} Window{currentAnalysis.n_windows > 1 ? 's' : ''} Analyzed
                </span>
              </div>
              <h1 className="text-2xl font-bold tracking-tight text-white">
                Prediction Results: {currentAnalysis.source_name}
              </h1>
            </div>

            <div className="flex items-center gap-3">
              <button
                onClick={handleExportCsv}
                className="flex items-center gap-2 text-xs font-semibold text-slate-200 bg-[#111827] border border-slate-800 hover:border-slate-700 hover:text-white px-3.5 py-2 rounded-lg transition-colors shadow-sm"
              >
                <Download className="w-4 h-4 text-blue-400" />
                <span>Export CSV Results</span>
              </button>
              <Link
                to="/analyzer"
                className="text-xs font-semibold text-white bg-blue-600 hover:bg-blue-500 px-3.5 py-2 rounded-lg transition-colors shadow-sm"
              >
                New Analysis
              </Link>
            </div>
          </div>

          {/* Warnings Banner if any */}
          {currentAnalysis.warnings && currentAnalysis.warnings.length > 0 && (
            <div className="bg-amber-950/50 border border-amber-800/80 rounded-xl p-4 flex items-start gap-3 text-amber-300 text-xs">
              <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
              <div className="space-y-1">
                <span className="font-semibold text-amber-200">Preprocessing Notes:</span>
                <ul className="list-disc list-inside space-y-0.5 text-amber-300/90">
                  {currentAnalysis.warnings.map((w, idx) => (
                    <li key={idx}>{w}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {/* Primary KPI Metrics */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div className="bg-[#111827] border border-slate-800 rounded-xl p-4">
              <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 mb-1">
                Dominant Activity
              </div>
              <div className="mt-1">
                <ActivityBadge
                  activity={currentAnalysis.dominant_activity}
                  label={currentAnalysis.dominant_label}
                />
              </div>
              {currentAnalysis.ground_truth && (
                <div className="text-[11px] text-slate-400 mt-2 font-mono">
                  Ground Truth: {currentAnalysis.ground_truth_label}
                </div>
              )}
            </div>

            <div className="bg-[#111827] border border-slate-800 rounded-xl p-4">
              <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 mb-1">
                Mean Confidence
              </div>
              <div className="text-2xl font-bold text-white tracking-tight">
                {(currentAnalysis.mean_confidence * 100).toFixed(1)}%
              </div>
              <div className="text-[11px] text-slate-500 mt-1">Softmax probability</div>
            </div>

            <div className="bg-[#111827] border border-slate-800 rounded-xl p-4">
              <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 mb-1">
                Model Agreement
              </div>
              <div className="text-2xl font-bold text-emerald-400 tracking-tight">
                {currentAnalysis.model_agreement_rate.toFixed(1)}%
              </div>
              <div className="text-[11px] text-slate-500 mt-1">Unanimous consensus</div>
            </div>

            <div className="bg-[#111827] border border-slate-800 rounded-xl p-4">
              <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 mb-1">
                Recording Duration
              </div>
              <div className="text-2xl font-bold text-white tracking-tight">
                {currentAnalysis.duration_sec.toFixed(2)}s
              </div>
              <div className="text-[11px] text-slate-500 mt-1">
                {currentAnalysis.n_windows} windows (Stride {currentAnalysis.stride})
              </div>
            </div>
          </div>

          {/* Activity Timeline (adapts cleanly for single-window) */}
          <TimelineChart
            windows={currentAnalysis.windows}
            durationSec={currentAnalysis.duration_sec}
          />

          {/* One clear confidence chart across windows */}
          {currentAnalysis.windows.length > 1 && (
            <ConfidenceChart windows={currentAnalysis.windows} />
          )}

          {/* Window-by-Window Prediction Table */}
          <div className="bg-[#111827] border border-slate-800 rounded-xl p-5 shadow-sm space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
              <div>
                <h3 className="text-sm font-bold text-white tracking-wide">
                  Window-Level Model Predictions
                </h3>
                <p className="text-xs text-slate-400">
                  Classifications and confidence from 1D-CNN, GRU, and LSTM across 128-sample windows
                </p>
              </div>

              <span className="text-xs font-mono text-slate-400">
                {currentAnalysis.windows.length} window{currentAnalysis.windows.length > 1 ? 's' : ''} total
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs text-slate-300">
                <thead className="bg-[#0b0f17] text-slate-400 uppercase text-[10px] tracking-wider border-b border-slate-800">
                  <tr>
                    <th className="py-2.5 px-3">Win ID</th>
                    <th className="py-2.5 px-3">Interval</th>
                    <th className="py-2.5 px-3">Consensus Activity</th>
                    <th className="py-2.5 px-3">DeepConvLSTM</th>
                    <th className="py-2.5 px-3">1D-CNN</th>
                    <th className="py-2.5 px-3">GRU</th>
                    <th className="py-2.5 px-3">LSTM</th>
                    <th className="py-2.5 px-3">Agreement</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono">
                  {currentAnalysis.windows
                    .slice((currentPage - 1) * pageSize, currentPage * pageSize)
                    .map((win) => {
                      const convlstm = win.models.convlstm;
                      const cnn = win.models.cnn1d;
                      const gru = win.models.gru;
                      const lstm = win.models.lstm;

                      return (
                        <tr key={win.window_id} className="hover:bg-slate-800/30 transition-colors">
                          <td className="py-3 px-3 font-bold text-white">#{win.window_id}</td>
                          <td className="py-3 px-3 text-slate-400">
                            {win.start_time_sec.toFixed(2)}s – {win.end_time_sec.toFixed(2)}s
                          </td>
                          <td className="py-3 px-3 font-sans">
                            <ActivityBadge
                              activity={win.consensus_activity}
                              label={win.consensus_label}
                            />
                          </td>

                          {/* DeepConvLSTM */}
                          <td className="py-3 px-3 font-sans">
                            {convlstm ? (
                              <div>
                                <span className="font-semibold text-cyan-300">
                                  {convlstm.predicted_label}
                                </span>
                                <span className="block text-[10px] text-slate-500 font-mono">
                                  {(convlstm.confidence * 100).toFixed(1)}% conf
                                </span>
                              </div>
                            ) : (
                              <span className="text-slate-500">---</span>
                            )}
                          </td>

                          {/* CNN1D */}
                          <td className="py-3 px-3 font-sans">
                            {cnn ? (
                              <div>
                                <span className="font-semibold text-slate-200">
                                  {cnn.predicted_label}
                                </span>
                                <span className="block text-[10px] text-slate-500 font-mono">
                                  {(cnn.confidence * 100).toFixed(1)}% conf
                                </span>
                              </div>
                            ) : (
                              <span className="text-slate-500">---</span>
                            )}
                          </td>

                          {/* GRU */}
                          <td className="py-3 px-3 font-sans">
                            {gru ? (
                              <div>
                                <span className="font-semibold text-slate-200">
                                  {gru.predicted_label}
                                </span>
                                <span className="block text-[10px] text-slate-500 font-mono">
                                  {(gru.confidence * 100).toFixed(1)}% conf
                                </span>
                              </div>
                            ) : (
                              <span className="text-slate-500">---</span>
                            )}
                          </td>

                          {/* LSTM */}
                          <td className="py-3 px-3 font-sans">
                            {lstm ? (
                              <div>
                                <span className="font-semibold text-slate-200">
                                  {lstm.predicted_label}
                                </span>
                                <span className="block text-[10px] text-slate-500 font-mono">
                                  {(lstm.confidence * 100).toFixed(1)}% conf
                                </span>
                              </div>
                            ) : (
                              <span className="text-slate-500">---</span>
                            )}
                          </td>

                          {/* Agreement */}
                          <td className="py-3 px-3 font-sans">
                            {win.unanimous_agreement ? (
                              <span className="inline-flex items-center gap-1 text-[11px] font-semibold text-emerald-400">
                                <CheckCircle2 className="w-3.5 h-3.5" />
                                <span>Unanimous</span>
                              </span>
                            ) : (
                              <span className="inline-flex items-center gap-1 text-[11px] font-semibold text-amber-400">
                                <AlertTriangle className="w-3.5 h-3.5" />
                                <span>Disagreement</span>
                              </span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                </tbody>
              </table>
            </div>

            {/* Pagination */}
            {currentAnalysis.windows.length > pageSize && (
              <div className="flex items-center justify-between pt-3 border-t border-slate-800">
                <span className="text-xs text-slate-500 font-mono">
                  Showing {(currentPage - 1) * pageSize + 1}–
                  {Math.min(currentPage * pageSize, currentAnalysis.windows.length)} of{' '}
                  {currentAnalysis.windows.length}
                </span>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setCurrentPage((p) => Math.max(p - 1, 1))}
                    disabled={currentPage === 1}
                    className="px-2.5 py-1 text-xs rounded border border-slate-800 bg-[#0b0f17] text-slate-300 disabled:opacity-40 hover:bg-slate-800 transition-colors"
                  >
                    <ChevronLeft className="w-3.5 h-3.5" />
                  </button>
                  <button
                    onClick={() =>
                      setCurrentPage((p) =>
                        Math.min(p + 1, Math.ceil(currentAnalysis.windows.length / pageSize))
                      )
                    }
                    disabled={currentPage === Math.ceil(currentAnalysis.windows.length / pageSize)}
                    className="px-2.5 py-1 text-xs rounded border border-slate-800 bg-[#0b0f17] text-slate-300 disabled:opacity-40 hover:bg-slate-800 transition-colors"
                  >
                    <ChevronRight className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      ) : (
        /* Empty Analysis State */
        <div className="bg-[#111827] border border-slate-800 rounded-2xl p-8 text-center space-y-3">
          <div className="w-12 h-12 rounded-xl bg-blue-950/60 border border-blue-800/80 text-blue-400 flex items-center justify-center mx-auto">
            <FileSpreadsheet className="w-6 h-6" />
          </div>
          <h2 className="text-base font-bold text-white">No Active Recording Analyzed Yet</h2>
          <p className="text-xs text-slate-400 max-w-md mx-auto">
            Select a benchmark sample or upload your custom accelerometer/gyroscope CSV file in the Activity Analyzer to view per-window deep learning classifications.
          </p>
          <div className="pt-2">
            <Link
              to="/analyzer"
              className="inline-flex items-center gap-2 text-xs font-semibold text-white bg-blue-600 hover:bg-blue-500 px-4 py-2 rounded-lg transition-colors shadow-sm"
            >
              <ArrowLeft className="w-4 h-4" />
              <span>Go to Activity Analyzer</span>
            </Link>
          </div>
        </div>
      )}

      {/* ── Section B: Consolidated Benchmark Model Evaluation & Confusion Matrix ── */}
      <div className="pt-4 border-t border-slate-800 space-y-6">
        <div>
          <h2 className="text-lg font-bold text-white tracking-wide">
            Model Evaluation & Benchmark Comparison
          </h2>
          <p className="text-xs text-slate-400">
            Performance comparison and empirical confusion matrix across the 2,947 UCI HAR test windows.
          </p>
        </div>

        {/* Model Performance Comparison Table */}
        {models.length > 0 && (
          <div className="bg-[#111827] border border-slate-800 rounded-xl p-5 shadow-sm space-y-3">
            <h3 className="text-sm font-bold text-white tracking-wide">
              Architecture Performance Overview
            </h3>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs text-slate-300">
                <thead className="bg-[#0b0f17] text-slate-400 uppercase text-[10px] tracking-wider border-b border-slate-800">
                  <tr>
                    <th className="py-2.5 px-3">Model</th>
                    <th className="py-2.5 px-3">Type</th>
                    <th className="py-2.5 px-3">Test Accuracy</th>
                    <th className="py-2.5 px-3">Macro-F1</th>
                    <th className="py-2.5 px-3">Parameters</th>
                    <th className="py-2.5 px-3">FP32 Size</th>
                    <th className="py-2.5 px-3">INT8 Edge</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono">
                  {models.map((m) => (
                    <tr key={m.id} className="hover:bg-slate-800/30 transition-colors">
                      <td className="py-2.5 px-3 font-sans font-semibold text-white">{m.name}</td>
                      <td className="py-2.5 px-3 font-sans text-slate-400">{m.type}</td>
                      <td className="py-2.5 px-3 font-bold text-emerald-400">
                        {m.test_accuracy !== null ? `${(m.test_accuracy * 100).toFixed(2)}%` : '---'}
                      </td>
                      <td className="py-2.5 px-3 text-blue-400">
                        {m.macro_f1 !== null ? `${(m.macro_f1 * 100).toFixed(2)}%` : '---'}
                      </td>
                      <td className="py-2.5 px-3 text-slate-300">
                        {m.parameters !== null ? m.parameters.toLocaleString() : 'N/A (Trees)'}
                      </td>
                      <td className="py-2.5 px-3 text-slate-400">
                        {m.disk_size_mb !== null ? `${m.disk_size_mb} MB` : 'N/A'}
                      </td>
                      <td className="py-2.5 px-3">
                        {m.int8_size_mb !== null ? (
                          <span className="text-cyan-300 font-bold bg-cyan-950/70 border border-cyan-800/60 px-1.5 py-0.5 rounded text-[11px]">
                            {m.int8_size_mb} MB
                          </span>
                        ) : (
                          <span className="text-slate-500">N/A</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Interactive Confusion Matrix */}
        {cmData && selectedMatrixData && (
          <div className="space-y-3">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
              <h3 className="text-sm font-bold text-white tracking-wide">
                Confusion Matrix (Error Analysis)
              </h3>

              <div className="flex items-center gap-1.5 bg-[#0b0f17] border border-slate-800 p-1 rounded-lg">
                {Object.keys(cmData.matrices).map((k) => (
                  <button
                    key={k}
                    onClick={() => setActiveCmModel(k)}
                    className={`text-xs px-2.5 py-1 rounded font-medium transition-colors ${
                      activeCmModel === k
                        ? 'bg-blue-600 text-white shadow-sm'
                        : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    {cmData.matrices[k].display_name}
                  </button>
                ))}
              </div>
            </div>

            <ConfusionMatrixHeatmap data={selectedMatrixData} />
          </div>
        )}
      </div>

      {/* Footer Disclaimer */}
      <div className="bg-[#0b0f17] border border-slate-800/80 rounded-xl p-4 flex items-start gap-3 text-xs text-slate-400">
        <ShieldCheck className="w-5 h-5 text-blue-400 shrink-0 mt-0.5" />
        <div>
          <span className="font-semibold text-slate-300">Evaluation Context: </span>
          Model confidence reflects internal softmax output probabilities on the 9-axis window tensor. Benchmark metrics are measured on the official 2,947-window test split and demonstrate model behavior within this project.
        </div>
      </div>
    </div>
  );
};
