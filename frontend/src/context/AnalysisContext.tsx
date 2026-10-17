import React, { createContext, useContext, useState } from 'react';
import type { PredictResponse } from '../types/api';

export interface AnalysisSessionRecord {
  id: string;
  timestamp: string;
  sourceName: string;
  durationSec: number;
  nWindows: number;
  dominantActivity: string;
  dominantLabel: string;
  agreementRate: number;
  meanConfidence: number;
  response: PredictResponse;
}

interface AnalysisContextType {
  currentAnalysis: PredictResponse | null;
  setCurrentAnalysis: (analysis: PredictResponse | null) => void;
  recentSessions: AnalysisSessionRecord[];
  clearCurrentAnalysis: () => void;
}

const AnalysisContext = createContext<AnalysisContextType | undefined>(undefined);

const CURRENT_ANALYSIS_STORAGE_KEY = 'motionsense_current_analysis';
const RECENT_SESSIONS_STORAGE_KEY = 'motionsense_recent_sessions';

export const AnalysisProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [currentAnalysis, setCurrentAnalysisState] = useState<PredictResponse | null>(() => {
    try {
      const stored = sessionStorage.getItem(CURRENT_ANALYSIS_STORAGE_KEY);
      return stored ? JSON.parse(stored) : null;
    } catch {
      return null;
    }
  });

  const [recentSessions, setRecentSessions] = useState<AnalysisSessionRecord[]>(() => {
    try {
      const stored = sessionStorage.getItem(RECENT_SESSIONS_STORAGE_KEY);
      return stored ? JSON.parse(stored) : [];
    } catch {
      return [];
    }
  });

  const setCurrentAnalysis = (analysis: PredictResponse | null) => {
    setCurrentAnalysisState(analysis);
    if (analysis) {
      try {
        sessionStorage.setItem(CURRENT_ANALYSIS_STORAGE_KEY, JSON.stringify(analysis));
      } catch {
        // Handle quota errors gracefully
      }

      // Add to session history without duplicate IDs
      const newRecord: AnalysisSessionRecord = {
        id: `${Date.now()}_${analysis.source_name}`,
        timestamp: new Date().toLocaleTimeString(),
        sourceName: analysis.source_name,
        durationSec: analysis.duration_sec,
        nWindows: analysis.n_windows,
        dominantActivity: analysis.dominant_activity,
        dominantLabel: analysis.dominant_label,
        agreementRate: analysis.model_agreement_rate,
        meanConfidence: analysis.mean_confidence,
        response: analysis,
      };

      setRecentSessions((prev) => {
        const updated = [newRecord, ...prev.slice(0, 9)];
        try {
          sessionStorage.setItem(RECENT_SESSIONS_STORAGE_KEY, JSON.stringify(updated));
        } catch {
          // ignore
        }
        return updated;
      });
    } else {
      sessionStorage.removeItem(CURRENT_ANALYSIS_STORAGE_KEY);
    }
  };

  const clearCurrentAnalysis = () => {
    setCurrentAnalysis(null);
  };

  return (
    <AnalysisContext.Provider
      value={{
        currentAnalysis,
        setCurrentAnalysis,
        recentSessions,
        clearCurrentAnalysis,
      }}
    >
      {children}
    </AnalysisContext.Provider>
  );
};

export function useAnalysis(): AnalysisContextType {
  const context = useContext(AnalysisContext);
  if (!context) {
    throw new Error('useAnalysis must be used within an AnalysisProvider');
  }
  return context;
}
