import React, { useState, useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useDropzone } from 'react-dropzone';
import {
  Upload,
  Cpu,
  Layers,
  ArrowRight,
  AlertCircle,
  CheckCircle2,
  FileText,
  Clock,
  Sparkles,
  HelpCircle,
  Camera,
} from 'lucide-react';
import { apiService } from '../api/client';
import type { SampleSummary, PredictResponse } from '../types/api';
import { ActivityBadge } from '../components/ActivityBadge';
import { useAnalysis } from '../context/AnalysisContext';

export const AnalyzerPage: React.FC = () => {
  const navigate = useNavigate();
  const { setCurrentAnalysis } = useAnalysis();

  // Mode Selection: 'benchmark' or 'custom'
  const [activeMode, setActiveMode] = useState<'benchmark' | 'custom'>('benchmark');

  // Mode 1: Benchmark Samples
  const [samples, setSamples] = useState<SampleSummary[]>([]);
  const [selectedSampleId, setSelectedSampleId] = useState<string>('');
  const [samplesLoading, setSamplesLoading] = useState(true);

  // Mode 2: Custom Upload
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);

  // Common Configuration: Stride
  const [stride, setStride] = useState<number>(64);

  // Execution State
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load benchmark samples on mount
  useEffect(() => {
    const fetchSamples = async () => {
      setSamplesLoading(true);
      try {
        const data = await apiService.getSamples();
        setSamples(data);
        if (data.length > 0) {
          setSelectedSampleId(data[0].id);
        }
      } catch (err: any) {
        setError(err?.detail || 'Failed to retrieve benchmark samples from backend.');
      } finally {
        setSamplesLoading(false);
      }
    };
    fetchSamples();
  }, []);

  // Dropzone configuration
  const { getRootProps, getInputProps, isDragActive, fileRejections } = useDropzone({
    accept: {
      'text/csv': ['.csv'],
    },
    maxFiles: 1,
    multiple: false,
    onDrop: (acceptedFiles) => {
      if (acceptedFiles.length > 0) {
        setUploadedFile(acceptedFiles[0]);
        setError(null);
      }
    },
  });

  const selectedSample = samples.find((s) => s.id === selectedSampleId);

  // Execute Analysis
  const handleRunInference = async () => {
    setIsAnalyzing(true);
    setError(null);

    try {
      let result: PredictResponse;

      if (activeMode === 'benchmark') {
        if (!selectedSampleId) {
          setError('Please select a benchmark sample to analyze.');
          setIsAnalyzing(false);
          return;
        }
        result = await apiService.predictSample(selectedSampleId, stride);
      } else {
        if (!uploadedFile) {
          setError('Please select or drop a compatible CSV recording.');
          setIsAnalyzing(false);
          return;
        }
        result = await apiService.predictCsv(uploadedFile, stride);
      }

      // Store in shared context and session storage
      setCurrentAnalysis(result);

      // Navigate to detailed results page
      navigate('/results');
    } catch (err: any) {
      setError(err?.detail || 'Inference execution failed. Check file structure and backend logs.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="space-y-8 max-w-5xl mx-auto">
      {/* Header */}
      <div className="pb-4 border-b border-slate-800">
        <div className="flex items-center gap-2 mb-1">
          <h1 className="text-2xl font-bold tracking-tight text-white">Activity Analyzer</h1>
          <span className="text-xs px-2 py-0.5 rounded-full font-mono bg-blue-950/80 text-blue-300 border border-blue-800/80">
            Multi-Model Inference
          </span>
        </div>
        <p className="text-sm text-slate-400">
          Run continuous forward inference across 1D-CNN, GRU, and LSTM on verified benchmark samples or custom sensor recordings.
        </p>
      </div>

      {/* Mode Switcher Tabs */}
      <div className="flex rounded-xl bg-[#0b0f17] border border-slate-800 p-1.5 gap-1.5">
        <button
          onClick={() => {
            setActiveMode('benchmark');
            setError(null);
          }}
          className={`flex-1 py-3 px-4 rounded-lg font-medium text-sm flex items-center justify-center gap-2 transition-all ${
            activeMode === 'benchmark'
              ? 'bg-blue-600 text-white shadow-md'
              : 'text-slate-400 hover:text-white hover:bg-slate-800/50'
          }`}
        >
          <Layers className="w-4 h-4" />
          <span>Mode 1: Benchmark Samples</span>
          <span className="text-[11px] font-mono px-1.5 py-0.2 rounded bg-black/30">
            {samples.length} Ready
          </span>
        </button>

        <button
          onClick={() => {
            setActiveMode('custom');
            setError(null);
          }}
          className={`flex-1 py-3 px-4 rounded-lg font-medium text-sm flex items-center justify-center gap-2 transition-all ${
            activeMode === 'custom'
              ? 'bg-blue-600 text-white shadow-md'
              : 'text-slate-400 hover:text-white hover:bg-slate-800/50'
          }`}
        >
          <Upload className="w-4 h-4" />
          <span>Mode 2: Custom CSV Upload</span>
          <span className="text-[11px] font-mono px-1.5 py-0.2 rounded bg-black/30">
            9-ch / 6-ch IMU
          </span>
        </button>

        <Link
          to="/vision"
          className="flex-1 py-3 px-4 rounded-lg font-medium text-sm flex items-center justify-center gap-2 transition-all text-slate-400 hover:text-white hover:bg-slate-800/50"
        >
          <Camera className="w-4 h-4 text-indigo-400" />
          <span>Mode 3: Camera & Photo AI</span>
          <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
            NEW
          </span>
        </Link>
      </div>

      {/* Error Alert */}
      {error && (
        <div className="bg-rose-950/70 border border-rose-800 rounded-xl p-4 flex items-start gap-3 text-rose-300 text-sm">
          <AlertCircle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
          <div>
            <div className="font-semibold text-rose-200">Analysis Error</div>
            <p className="mt-0.5 text-xs text-rose-300/90">{error}</p>
          </div>
        </div>
      )}

      {/* Mode 1: Benchmark Samples Content */}
      {activeMode === 'benchmark' && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-bold text-white tracking-wide uppercase">
              Select Verified Benchmark Sample
            </h2>
            <span className="text-xs text-slate-400">
              Source: UCI HAR Test Partition (50 Hz)
            </span>
          </div>

          {samplesLoading ? (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {[1, 2, 3].map((i) => (
                <div key={i} className="h-36 bg-[#111827] rounded-xl border border-slate-800 animate-pulse" />
              ))}
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {samples.map((s) => {
                const isSelected = selectedSampleId === s.id;
                return (
                  <div
                    key={s.id}
                    onClick={() => setSelectedSampleId(s.id)}
                    className={`p-4 rounded-xl border transition-all cursor-pointer flex flex-col justify-between ${
                      isSelected
                        ? 'bg-blue-950/20 border-blue-500 shadow-md ring-1 ring-blue-500/40'
                        : 'bg-[#111827] border-slate-800 hover:border-slate-700'
                    }`}
                  >
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-xs font-mono text-slate-400">{s.filename}</span>
                        {s.ground_truth && (
                          <ActivityBadge
                            activity={s.ground_truth}
                            label={s.ground_truth_label || undefined}
                          />
                        )}
                      </div>
                      <h3 className="text-sm font-bold text-white mb-1">{s.name}</h3>
                      <p className="text-xs text-slate-400 line-clamp-2">{s.description}</p>
                    </div>

                    <div className="mt-4 pt-3 border-t border-slate-800/80 flex items-center justify-between text-[11px] text-slate-400 font-mono">
                      <span>{s.expected_rows} rows ({s.duration_sec.toFixed(2)}s)</span>
                      <span>{s.channels} Channels</span>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}

      {/* Mode 2: Custom CSV Upload Content */}
      {activeMode === 'custom' && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-bold text-white tracking-wide uppercase">
              Upload Wearable Inertial Sensor CSV
            </h2>
            <span className="text-xs text-slate-400">
              50 Hz Sampling Rate · Minimum 128 rows (2.56s)
            </span>
          </div>

          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors ${
              isDragActive
                ? 'border-blue-500 bg-blue-950/20'
                : uploadedFile
                ? 'border-emerald-600 bg-emerald-950/10'
                : 'border-slate-800 hover:border-slate-700 bg-[#111827]'
            }`}
          >
            <input {...getInputProps()} />
            <div className="w-12 h-12 rounded-xl bg-slate-800 text-blue-400 flex items-center justify-center mx-auto mb-3">
              <Upload className="w-6 h-6" />
            </div>

            {uploadedFile ? (
              <div>
                <div className="flex items-center justify-center gap-2 text-emerald-400 font-semibold text-sm mb-1">
                  <CheckCircle2 className="w-4 h-4" />
                  <span>Selected: {uploadedFile.name}</span>
                </div>
                <p className="text-xs text-slate-400 font-mono">
                  {(uploadedFile.size / 1024).toFixed(1)} KB · Ready for server-side Butterworth filtering & inference
                </p>
                <p className="text-[11px] text-slate-500 mt-2">
                  Click or drag another file to replace
                </p>
              </div>
            ) : (
              <div>
                <p className="text-sm font-semibold text-white mb-1">
                  Drop your smartphone sensor recording here, or browse files
                </p>
                <p className="text-xs text-slate-400 max-w-md mx-auto">
                  Supports <strong>9-channel full standard</strong> (body acc, gyro, total acc) or <strong>6-channel raw IMU</strong> (total acc + gyro).
                </p>
              </div>
            )}
          </div>

          {fileRejections.length > 0 && (
            <div className="text-xs text-rose-400 flex items-center gap-1.5">
              <AlertCircle className="w-4 h-4" />
              <span>Only standard comma-separated text files (.csv) are accepted.</span>
            </div>
          )}

          {/* Schema Help Box */}
          <div className="bg-[#0b0f17] border border-slate-800 rounded-xl p-4 text-xs space-y-2">
            <div className="flex items-center gap-2 text-slate-300 font-semibold">
              <HelpCircle className="w-4 h-4 text-blue-400" />
              <span>Supported Input Formats:</span>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-slate-400">
              <div className="bg-[#111827] p-2.5 rounded-lg border border-slate-800/80">
                <span className="block font-semibold text-slate-200 mb-1">Option A: 9-Channel Standard</span>
                <code className="text-[10px] text-blue-300 block break-all">
                  body_acc_x,y,z · body_gyro_x,y,z · total_acc_x,y,z
                </code>
              </div>
              <div className="bg-[#111827] p-2.5 rounded-lg border border-slate-800/80">
                <span className="block font-semibold text-slate-200 mb-1">Option B: 6-Channel Raw IMU</span>
                <code className="text-[10px] text-emerald-300 block break-all">
                  total_acc_x,y,z · body_gyro_x,y,z (Auto-Butterworth 0.3 Hz gravity separation)
                </code>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Sliding Window Stride Selector */}
      <div className="bg-[#111827] border border-slate-800 rounded-xl p-5 shadow-sm space-y-3">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-bold text-white tracking-wide">
              Sliding Window Segmentation Stride
            </h3>
            <p className="text-xs text-slate-400">
              Window size is fixed to 128 timesteps (2.56 seconds at 50 Hz). Choose temporal stepping rate:
            </p>
          </div>
          <span className="text-xs font-mono text-slate-400">
            Window: 128 samples
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <label
            onClick={() => setStride(64)}
            className={`p-3.5 rounded-lg border cursor-pointer flex items-center justify-between transition-colors ${
              stride === 64
                ? 'bg-blue-950/30 border-blue-500 text-white'
                : 'bg-[#0b0f17] border-slate-800 text-slate-400 hover:text-white'
            }`}
          >
            <div className="flex items-center gap-3">
              <input
                type="radio"
                name="stride"
                value="64"
                checked={stride === 64}
                onChange={() => setStride(64)}
                className="text-blue-600 focus:ring-blue-500"
              />
              <div>
                <div className="text-xs font-bold text-white">64 Samples (50% Overlap)</div>
                <div className="text-[11px] text-slate-400">
                  Step every 1.28s. Recommended for continuous activity transitions.
                </div>
              </div>
            </div>
            <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-blue-950 text-blue-300 border border-blue-800">
              Benchmark Standard
            </span>
          </label>

          <label
            onClick={() => setStride(128)}
            className={`p-3.5 rounded-lg border cursor-pointer flex items-center justify-between transition-colors ${
              stride === 128
                ? 'bg-blue-950/30 border-blue-500 text-white'
                : 'bg-[#0b0f17] border-slate-800 text-slate-400 hover:text-white'
            }`}
          >
            <div className="flex items-center gap-3">
              <input
                type="radio"
                name="stride"
                value="128"
                checked={stride === 128}
                onChange={() => setStride(128)}
                className="text-blue-600 focus:ring-blue-500"
              />
              <div>
                <div className="text-xs font-bold text-white">128 Samples (Non-Overlapping)</div>
                <div className="text-[11px] text-slate-400">
                  Step every 2.56s. Faster execution, discrete isolated blocks.
                </div>
              </div>
            </div>
            <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
              Discrete Windows
            </span>
          </label>
        </div>
      </div>

      {/* Action Footer */}
      <div className="bg-[#0b0f17] border border-slate-800 rounded-xl p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="text-xs text-slate-400 flex items-center gap-2">
          <Clock className="w-4 h-4 text-slate-500 shrink-0" />
          <span>
            {activeMode === 'benchmark'
              ? `Selected: ${selectedSample?.name || 'None'} (Stride: ${stride})`
              : `Uploaded: ${uploadedFile?.name || 'No file selected'} (Stride: ${stride})`}
          </span>
        </div>

        <button
          onClick={handleRunInference}
          disabled={isAnalyzing || (activeMode === 'benchmark' && !selectedSampleId) || (activeMode === 'custom' && !uploadedFile)}
          className={`flex items-center justify-center gap-2 px-6 py-2.5 rounded-lg font-semibold text-sm transition-all shadow-md ${
            isAnalyzing
              ? 'bg-blue-700 text-white cursor-not-allowed opacity-80'
              : 'bg-blue-600 hover:bg-blue-500 text-white'
          }`}
        >
          {isAnalyzing ? (
            <>
              <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              <span>Executing Deep Models...</span>
            </>
          ) : (
            <>
              <Sparkles className="w-4 h-4" />
              <span>Execute Activity Inference</span>
              <ArrowRight className="w-4 h-4" />
            </>
          )}
        </button>
      </div>
    </div>
  );
};
