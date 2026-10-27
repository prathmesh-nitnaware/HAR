import axios, { AxiosError } from 'axios';
import type {
  HealthResponse,
  ModelMetadata,
  SampleSummary,
  SampleDetailResponse,
  PredictResponse,
  MetricsResponse,
  ConfusionMatrixResponse,
  ApiError,
  VisionPredictionResponse,
  VisionDemoSamplesResponse,
  NetworkInfoResponse,
} from '../types/api';

export const getBaseUrl = (): string => {
  if (import.meta.env.VITE_API_BASE_URL) {
    return import.meta.env.VITE_API_BASE_URL;
  }
  if (typeof window !== 'undefined') {
    // In the browser, an empty string ensures requests are relative to current origin,
    // which Vite proxies seamlessly (via vite.config.ts) from port 5173 to port 8000.
    // This avoids CORS, firewall blocks, and https/http mixed content issues on phones.
    return '';
  }
  return 'http://127.0.0.1:8000';
};

const BASE_URL = getBaseUrl();

export const apiClient = axios.create({
  baseURL: BASE_URL,
  timeout: 30000,
});

export function parseApiError(error: unknown): ApiError {
  if (axios.isAxiosError(error)) {
    const axiosErr = error as AxiosError<{ detail?: string; error_type?: string }>;
    if (axiosErr.response?.data?.detail) {
      return {
        detail: axiosErr.response.data.detail,
        error_type: axiosErr.response.data.error_type || 'ApiError',
        status: axiosErr.response.status,
      };
    }
    if (axiosErr.code === 'ECONNABORTED') {
      return {
        detail: 'Request timed out while waiting for server response.',
        error_type: 'TimeoutError',
      };
    }
    if (!axiosErr.response) {
      return {
        detail: `Cannot connect to server. Ensure backend and dev server are running.`,
        error_type: 'NetworkError',
      };
    }
    return {
      detail: axiosErr.message || 'An unexpected API error occurred.',
      status: axiosErr.response.status,
    };
  }
  return {
    detail: (error as Error)?.message || 'An unexpected error occurred.',
    error_type: 'UnknownError',
  };
}

export const apiService = {
  async getHealth(): Promise<HealthResponse> {
    try {
      const response = await apiClient.get<HealthResponse>('/api/health');
      return response.data;
    } catch (err) {
      throw parseApiError(err);
    }
  },

  async getModels(): Promise<ModelMetadata[]> {
    try {
      const response = await apiClient.get<ModelMetadata[]>('/api/models');
      return response.data;
    } catch (err) {
      throw parseApiError(err);
    }
  },

  async getSamples(): Promise<SampleSummary[]> {
    try {
      const response = await apiClient.get<SampleSummary[]>('/api/samples');
      return response.data;
    } catch (err) {
      throw parseApiError(err);
    }
  },

  async getSampleDetail(sampleId: string): Promise<SampleDetailResponse> {
    try {
      const response = await apiClient.get<SampleDetailResponse>(`/api/samples/${sampleId}`);
      return response.data;
    } catch (err) {
      throw parseApiError(err);
    }
  },

  async predictSample(sampleId: string, stride: number = 128): Promise<PredictResponse> {
    try {
      const response = await apiClient.post<PredictResponse>(
        `/api/predict/sample/${sampleId}`,
        null,
        {
          params: { stride },
        }
      );
      return response.data;
    } catch (err) {
      throw parseApiError(err);
    }
  },

  async predictCsv(file: File, stride: number = 64): Promise<PredictResponse> {
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('stride', String(stride));

      const response = await apiClient.post<PredictResponse>('/api/predict/csv', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      return response.data;
    } catch (err) {
      throw parseApiError(err);
    }
  },

  async getMetrics(): Promise<MetricsResponse> {
    try {
      const response = await apiClient.get<MetricsResponse>('/api/metrics');
      return response.data;
    } catch (err) {
      throw parseApiError(err);
    }
  },

  async getConfusionMatrices(): Promise<ConfusionMatrixResponse> {
    try {
      const response = await apiClient.get<ConfusionMatrixResponse>('/api/confusion-matrices');
      return response.data;
    } catch (err) {
      throw parseApiError(err);
    }
  },

  async getSimulationSamples(): Promise<Record<string, { name: string; activity: string; readings: any[] }>> {
    try {
      const response = await apiClient.get<{ samples: Record<string, { name: string; activity: string; readings: any[] }> }>('/api/stream/simulation-samples');
      return response.data.samples;
    } catch (err) {
      throw parseApiError(err);
    }
  },

  async getVisionDemoSamples(): Promise<VisionDemoSamplesResponse> {
    try {
      const response = await apiClient.get<VisionDemoSamplesResponse>('/api/vision/demo-samples');
      return response.data;
    } catch (err) {
      throw parseApiError(err);
    }
  },

  async predictVisionFile(file: File): Promise<VisionPredictionResponse> {
    try {
      const formData = new FormData();
      formData.append('file', file);
      const response = await apiClient.post<VisionPredictionResponse>('/api/vision/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      return response.data;
    } catch (err) {
      throw parseApiError(err);
    }
  },

  async predictVisionBase64(imageBase64: string): Promise<VisionPredictionResponse> {
    try {
      const response = await apiClient.post<VisionPredictionResponse>('/api/vision/predict-base64', {
        image_base64: imageBase64,
      });
      return response.data;
    } catch (err) {
      throw parseApiError(err);
    }
  },

  async getNetworkInfo(): Promise<NetworkInfoResponse> {
    try {
      const response = await apiClient.get<NetworkInfoResponse>('/api/system/network-info');
      return response.data;
    } catch (err) {
      throw parseApiError(err);
    }
  },

  async uploadRemotePhoto(file: File): Promise<{
    status: string;
    update_id: string;
    predicted_activity: string;
    confidence: number;
    result: VisionPredictionResponse;
  }> {
    try {
      const formData = new FormData();
      formData.append('file', file);
      const response = await apiClient.post('/api/vision/remote-upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      return response.data;
    } catch (err) {
      throw parseApiError(err);
    }
  },

  async getLatestRemoteVision(): Promise<{
    update_id: string | null;
    timestamp: number;
    result: VisionPredictionResponse | null;
  }> {
    try {
      const response = await apiClient.get('/api/vision/latest-remote');
      return response.data;
    } catch (err) {
      throw parseApiError(err);
    }
  },

  getWebSocketUrl(): string {
    if (typeof window !== 'undefined') {
      const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      return `${proto}//${window.location.host}/ws/stream`;
    }
    return 'ws://127.0.0.1:8000/ws/stream';
  },
};


