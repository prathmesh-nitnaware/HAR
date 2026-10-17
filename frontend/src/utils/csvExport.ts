import type { PredictResponse } from '../types/api';

/**
 * Safely escape CSV string field according to RFC 4180
 */
export function escapeCsvField(val: string | number | boolean | null | undefined): string {
  if (val === null || val === undefined) return '';
  const str = String(val);
  if (str.includes(',') || str.includes('"') || str.includes('\n') || str.includes('\r')) {
    return `"${str.replace(/"/g, '""')}"`;
  }
  return str;
}

/**
 * Generate standard CSV from actual returned PredictResponse windows
 */
export function exportPredictionsToCsv(analysis: PredictResponse): string {
  const models = analysis.available_models;
  
  // Header row
  const headers = [
    'window_id',
    'start_time_sec',
    'end_time_sec',
    'consensus_activity',
    'consensus_label',
    'unanimous_agreement',
  ];

  // Dynamic headers for each available model
  for (const m of models) {
    headers.push(`${m}_prediction`);
    headers.push(`${m}_confidence`);
  }

  const rows: string[] = [headers.join(',')];

  for (const win of analysis.windows) {
    const rowValues = [
      escapeCsvField(win.window_id),
      escapeCsvField(win.start_time_sec.toFixed(3)),
      escapeCsvField(win.end_time_sec.toFixed(3)),
      escapeCsvField(win.consensus_activity),
      escapeCsvField(win.consensus_label),
      escapeCsvField(win.unanimous_agreement ? 'TRUE' : 'FALSE'),
    ];

    for (const m of models) {
      const modelPred = win.models[m];
      if (modelPred) {
        rowValues.push(escapeCsvField(modelPred.predicted_label));
        rowValues.push(escapeCsvField((modelPred.confidence * 100).toFixed(2) + '%'));
      } else {
        rowValues.push('');
        rowValues.push('');
      }
    }

    rows.push(rowValues.join(','));
  }

  return rows.join('\r\n');
}

/**
 * Trigger client-side browser file download
 */
export function downloadCsvFile(csvContent: string, filename: string): void {
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.setAttribute('href', url);
  link.setAttribute('download', filename);
  link.style.visibility = 'hidden';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}
