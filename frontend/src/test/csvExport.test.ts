import { describe, it, expect } from 'vitest';
import { escapeCsvField, exportPredictionsToCsv } from '../utils/csvExport';
import type { PredictResponse } from '../types/api';

describe('CSV Export Utility', () => {
  it('correctly escapes CSV fields containing commas or quotes', () => {
    expect(escapeCsvField('hello,world')).toBe('"hello,world"');
    expect(escapeCsvField('hello"world')).toBe('"hello""world"');
    expect(escapeCsvField('normal')).toBe('normal');
    expect(escapeCsvField(42)).toBe('42');
    expect(escapeCsvField(null)).toBe('');
  });

  it('generates well-formed CSV with per-model predictions and confidences', () => {
    const mockAnalysis: PredictResponse = {
      source_name: 'test_sample.csv',
      duration_sec: 2.56,
      n_rows: 128,
      n_windows: 1,
      stride: 128,
      schema_detected: '9-Channel Full',
      warnings: [],
      dominant_activity: 'walking',
      dominant_label: 'Walking',
      model_agreement_rate: 100.0,
      mean_confidence: 0.999,
      available_models: ['cnn1d', 'gru', 'lstm'],
      windows: [
        {
          window_id: 1,
          start_time_sec: 0.0,
          end_time_sec: 2.56,
          consensus_activity: 'walking',
          consensus_label: 'Walking',
          unanimous_agreement: true,
          models: {
            cnn1d: {
              predicted_activity: 'walking',
              predicted_label: 'Walking',
              confidence: 0.999,
              probabilities: { walking: 0.999 },
            },
            gru: {
              predicted_activity: 'walking',
              predicted_label: 'Walking',
              confidence: 0.985,
              probabilities: { walking: 0.985 },
            },
            lstm: {
              predicted_activity: 'walking',
              predicted_label: 'Walking',
              confidence: 0.99,
              probabilities: { walking: 0.99 },
            },
          },
        },
      ],
    };

    const csvOutput = exportPredictionsToCsv(mockAnalysis);
    const lines = csvOutput.split('\r\n');

    expect(lines.length).toBe(2);
    expect(lines[0]).toContain('window_id,start_time_sec,end_time_sec,consensus_activity,consensus_label,unanimous_agreement');
    expect(lines[0]).toContain('cnn1d_prediction,cnn1d_confidence');
    expect(lines[1]).toContain('1,0.000,2.560,walking,Walking,TRUE');
    expect(lines[1]).toContain('Walking,99.90%');
  });
});
