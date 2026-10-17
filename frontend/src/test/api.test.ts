import { describe, it, expect, vi, beforeEach } from 'vitest';
import { apiService, apiClient, parseApiError } from '../api/client';

describe('API Client and Service Functions', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('fetches health status successfully', async () => {
    const mockHealth = {
      status: 'healthy',
      device: 'cpu',
      models: { cnn1d: true, gru: true, lstm: true },
      models_loaded_count: 3,
      total_models: 3,
    };

    vi.spyOn(apiClient, 'get').mockResolvedValueOnce({ data: mockHealth });

    const result = await apiService.getHealth();
    expect(result.status).toBe('healthy');
    expect(result.models_loaded_count).toBe(3);
    expect(apiClient.get).toHaveBeenCalledWith('/api/health');
  });

  it('fetches model registry correctly', async () => {
    const mockModels = [
      {
        id: 'cnn1d',
        name: '1D-CNN',
        type: 'Deep ConvNet',
        architecture: '3 Conv Blocks',
        input_representation: 'Raw 9x128',
        parameters: 709830,
        disk_size_mb: 2.73,
        status: 'ready',
        test_accuracy: 0.942,
        macro_f1: 0.9433,
      },
    ];

    vi.spyOn(apiClient, 'get').mockResolvedValueOnce({ data: mockModels });

    const result = await apiService.getModels();
    expect(result.length).toBe(1);
    expect(result[0].id).toBe('cnn1d');
    expect(apiClient.get).toHaveBeenCalledWith('/api/models');
  });

  it('correctly passes stride to predictSample endpoint', async () => {
    const mockPred = {
      source_name: 'sample_walking_128.csv',
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
      available_models: ['cnn1d'],
      windows: [],
    };

    vi.spyOn(apiClient, 'post').mockResolvedValueOnce({ data: mockPred });

    const result = await apiService.predictSample('walking_128', 128);
    expect(result.source_name).toBe('sample_walking_128.csv');
    expect(apiClient.post).toHaveBeenCalledWith(
      '/api/predict/sample/walking_128',
      null,
      { params: { stride: 128 } }
    );
  });

  it('parses network and HTTP errors accurately', () => {
    const genericErr = new Error('Network failure');
    const parsed = parseApiError(genericErr);
    expect(parsed.detail).toBe('Network failure');
  });
});
