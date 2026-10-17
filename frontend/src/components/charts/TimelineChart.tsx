import React from 'react';
import type { WindowPrediction } from '../../types/api';
import { ActivityBadge } from '../ActivityBadge';
import { CheckCircle2, AlertTriangle } from 'lucide-react';

interface TimelineChartProps {
  windows: WindowPrediction[];
  durationSec: number;
}

export const TimelineChart: React.FC<TimelineChartProps> = ({ windows, durationSec }) => {
  if (!windows || windows.length === 0) {
    return (
      <div className="bg-[#111827] border border-slate-800 rounded-xl p-5 text-center text-slate-400 text-xs">
        No window predictions available.
      </div>
    );
  }

  // Handle single-window recording cleanly
  if (windows.length === 1) {
    const win = windows[0];
    return (
      <div className="bg-[#111827] border border-slate-800 rounded-xl p-5">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-bold text-white tracking-wide">
            Window Prediction Timeline (Single Window)
          </h3>
          <span className="text-xs font-mono text-slate-400">
            0.00s – {win.end_time_sec.toFixed(2)}s (128 samples @ 50 Hz)
          </span>
        </div>

        <div className="bg-[#0b0f17] border border-slate-800 rounded-lg p-4 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-blue-950/80 border border-blue-800/80 flex items-center justify-center font-bold text-blue-400">
              #1
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-sm font-semibold text-white">Consensus Activity:</span>
                <ActivityBadge activity={win.consensus_activity} label={win.consensus_label} />
              </div>
              <p className="text-xs text-slate-400 mt-1">
                Duration: {win.start_time_sec.toFixed(2)}s to {win.end_time_sec.toFixed(2)}s
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {win.unanimous_agreement ? (
              <span className="inline-flex items-center gap-1.5 text-xs font-medium text-emerald-400 bg-emerald-950/70 border border-emerald-800/60 px-3 py-1.5 rounded-lg">
                <CheckCircle2 className="w-4 h-4" /> Unanimous Model Agreement
              </span>
            ) : (
              <span className="inline-flex items-center gap-1.5 text-xs font-medium text-amber-400 bg-amber-950/70 border border-amber-800/60 px-3 py-1.5 rounded-lg">
                <AlertTriangle className="w-4 h-4" /> Architecture Disagreement
              </span>
            )}
          </div>
        </div>
      </div>
    );
  }

  // Multi-window timeline visualizer
  return (
    <div className="bg-[#111827] border border-slate-800 rounded-xl p-5 shadow-sm">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-4">
        <div>
          <h3 className="text-sm font-bold text-white tracking-wide">
            Temporal Activity Timeline Across Windows
          </h3>
          <p className="text-xs text-slate-400">
            {windows.length} sequential windows over {durationSec.toFixed(2)} seconds
          </p>
        </div>
        <div className="flex items-center gap-3 text-xs text-slate-400">
          <span className="flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded-sm bg-blue-500 inline-block" /> Walking
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded-sm bg-indigo-500 inline-block" /> Upstairs
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded-sm bg-cyan-500 inline-block" /> Downstairs
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded-sm bg-amber-500 inline-block" /> Sitting
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded-sm bg-purple-500 inline-block" /> Standing
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded-sm bg-emerald-500 inline-block" /> Laying
          </span>
        </div>
      </div>

      {/* Progress timeline bar */}
      <div className="space-y-3">
        <div className="h-9 w-full bg-[#0b0f17] border border-slate-800 rounded-lg overflow-hidden flex p-1 gap-1">
          {windows.map((w) => {
            const widthPct = Math.max((1 / windows.length) * 100, 3);
            const colorClass =
              w.consensus_activity === 'walking'
                ? 'bg-blue-600 hover:bg-blue-500'
                : w.consensus_activity === 'walking-upstairs'
                ? 'bg-indigo-600 hover:bg-indigo-500'
                : w.consensus_activity === 'walking-downstairs'
                ? 'bg-cyan-600 hover:bg-cyan-500'
                : w.consensus_activity === 'sitting'
                ? 'bg-amber-600 hover:bg-amber-500'
                : w.consensus_activity === 'standing'
                ? 'bg-purple-600 hover:bg-purple-500'
                : 'bg-emerald-600 hover:bg-emerald-500';

            return (
              <div
                key={`timeline-bar-${w.window_id}`}
                style={{ width: `${widthPct}%` }}
                className={`h-full rounded transition-all cursor-pointer relative group ${colorClass}`}
                title={`Window #${w.window_id}: ${w.consensus_label} (${w.start_time_sec.toFixed(2)}s - ${w.end_time_sec.toFixed(2)}s)`}
              >
                {!w.unanimous_agreement && (
                  <span className="absolute top-0 right-0 w-2 h-2 rounded-full bg-amber-400 border border-slate-900" />
                )}
              </div>
            );
          })}
        </div>

        <div className="flex justify-between text-[11px] font-mono text-slate-500 px-1">
          <span>0.00s</span>
          <span>{(durationSec / 2).toFixed(2)}s</span>
          <span>{durationSec.toFixed(2)}s</span>
        </div>
      </div>
    </div>
  );
};
