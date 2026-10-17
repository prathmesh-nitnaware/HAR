import React, { useState } from 'react';
import type { ConfusionMatrixData } from '../../types/api';

interface ConfusionMatrixHeatmapProps {
  data: ConfusionMatrixData;
}

export const ConfusionMatrixHeatmap: React.FC<ConfusionMatrixHeatmapProps> = ({ data }) => {
  const [showNormalized, setShowNormalized] = useState(true);

  const activities = data.activities;
  const labels = data.activity_labels;
  const rawMatrix = data.matrix;
  const normMatrix = data.matrix_normalized;

  const activeMatrix = showNormalized ? normMatrix : rawMatrix;

  // Calculate cell color based on normalized value (0.0 to 1.0)
  const getCellColor = (i: number, j: number) => {
    const val = normMatrix[i][j];
    const isDiagonal = i === j;

    if (isDiagonal) {
      // Emerald / Blue gradient for true positives
      if (val >= 0.95) return 'bg-emerald-600/90 text-white font-bold';
      if (val >= 0.85) return 'bg-emerald-700/80 text-white font-semibold';
      if (val >= 0.70) return 'bg-emerald-800/70 text-emerald-100';
      return 'bg-emerald-900/60 text-emerald-200';
    } else {
      // Amber / Rose alert for off-diagonal errors
      if (val > 0.15) return 'bg-rose-950/90 text-rose-300 font-bold border border-rose-800';
      if (val > 0.05) return 'bg-amber-950/80 text-amber-300 font-medium';
      if (val > 0.01) return 'bg-amber-950/40 text-amber-400/80';
      return 'bg-slate-900/50 text-slate-500';
    }
  };

  return (
    <div className="bg-[#111827] border border-slate-800 rounded-xl p-5 shadow-sm">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-5">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="text-base font-bold text-white tracking-wide">
              {data.display_name} Confusion Matrix
            </h3>
            <span className="text-xs font-mono px-2 py-0.5 rounded bg-blue-950/80 text-blue-300 border border-blue-800/80">
              Evaluated on 2,947 Windows
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            <strong>Rows</strong> indicate <em>Ground-Truth (Actual)</em> classes; <strong>Columns</strong> indicate <em>Model Predicted</em> classes.
          </p>
        </div>

        {/* View Mode Toggle */}
        <div className="flex items-center gap-1 bg-[#0b0f17] border border-slate-800 p-1 rounded-lg self-start">
          <button
            onClick={() => setShowNormalized(true)}
            className={`text-xs px-3 py-1 rounded font-medium transition-colors ${
              showNormalized
                ? 'bg-blue-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            Normalized (%)
          </button>
          <button
            onClick={() => setShowNormalized(false)}
            className={`text-xs px-3 py-1 rounded font-medium transition-colors ${
              !showNormalized
                ? 'bg-blue-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            Raw Counts
          </button>
        </div>
      </div>

      {/* Heatmap Grid */}
      <div className="overflow-x-auto pb-2">
        <div className="min-w-[620px]">
          {/* Column Headers (Predicted Labels) */}
          <div className="flex items-center mb-1">
            <div className="w-32 shrink-0 text-right pr-3 text-[11px] font-bold text-slate-400 uppercase tracking-wider">
              Actual \ Pred
            </div>
            <div className="flex-1 grid grid-cols-6 gap-1.5">
              {labels.map((lbl, idx) => (
                <div
                  key={`col-hdr-${idx}`}
                  className="text-center text-[11px] font-semibold text-slate-300 truncate px-1 py-1 bg-slate-900/60 rounded border border-slate-800"
                  title={`Predicted: ${lbl}`}
                >
                  {lbl}
                </div>
              ))}
            </div>
          </div>

          {/* Matrix Rows */}
          <div className="space-y-1.5">
            {labels.map((actualLabel, rowIdx) => (
              <div key={`row-${rowIdx}`} className="flex items-center">
                {/* Row Header (Actual Label) */}
                <div
                  className="w-32 shrink-0 text-right pr-3 text-xs font-medium text-slate-300 truncate"
                  title={`Actual Ground Truth: ${actualLabel}`}
                >
                  {actualLabel}
                </div>

                {/* Cells */}
                <div className="flex-1 grid grid-cols-6 gap-1.5">
                  {labels.map((predLabel, colIdx) => {
                    const rawVal = rawMatrix[rowIdx][colIdx];
                    const normVal = normMatrix[rowIdx][colIdx];
                    const displayVal = showNormalized
                      ? `${(normVal * 100).toFixed(1)}%`
                      : rawVal.toLocaleString();

                    return (
                      <div
                        key={`cell-${rowIdx}-${colIdx}`}
                        className={`h-11 rounded flex flex-col items-center justify-center text-xs transition-transform hover:scale-105 cursor-default relative group ${getCellColor(
                          rowIdx,
                          colIdx
                        )}`}
                        title={`Actual: ${actualLabel} -> Predicted: ${predLabel} | Count: ${rawVal} (${(normVal * 100).toFixed(2)}%)`}
                      >
                        <span>{displayVal}</span>
                        {showNormalized && (
                          <span className="text-[9px] opacity-75">{rawVal}</span>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="mt-4 pt-3 border-t border-slate-800/80 flex flex-wrap items-center justify-between text-xs text-slate-400 gap-2">
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded bg-emerald-600 inline-block" /> High Accuracy (&gt;90%)
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded bg-rose-950 border border-rose-800 inline-block" /> Significant Confusion (&gt;15%)
          </span>
        </div>
        <span className="text-slate-500 font-mono text-[11px]">
          Diagonal = True Positives · Off-Diagonal = Misclassifications
        </span>
      </div>
    </div>
  );
};
