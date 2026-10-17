import React, { useEffect, useState } from 'react';
import { Activity, Cpu, RefreshCw, AlertCircle, CheckCircle2, Menu } from 'lucide-react';
import { apiService } from '../api/client';
import type { HealthResponse } from '../types/api';

interface NavbarProps {
  onToggleSidebar: () => void;
  isSidebarOpen: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({ onToggleSidebar }) => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [offline, setOffline] = useState(false);

  const fetchHealth = async () => {
    setIsRefreshing(true);
    try {
      const data = await apiService.getHealth();
      setHealth(data);
      setOffline(false);
    } catch {
      setOffline(true);
      setHealth(null);
    } finally {
      setIsRefreshing(false);
    }
  };

  useEffect(() => {
    fetchHealth();
    const interval = setInterval(fetchHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="h-16 bg-[#0b0f17] border-b border-slate-800/80 px-4 sm:px-6 flex items-center justify-between sticky top-0 z-30">
      <div className="flex items-center gap-3">
        <button
          onClick={onToggleSidebar}
          aria-label="Toggle navigation sidebar"
          className="p-2 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800/70 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors"
        >
          <Menu className="w-5 h-5" />
        </button>

        <div className="flex items-center gap-2.5">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-blue-600 to-indigo-600 flex items-center justify-center shadow-md shadow-blue-500/20">
            <Activity className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-bold text-base tracking-tight text-white">MotionSense AI</span>
              <span className="text-[10px] uppercase font-bold tracking-widest px-1.5 py-0.5 rounded bg-blue-950/80 text-blue-400 border border-blue-800/60">
                HAR Deep Lab
              </span>
            </div>
            <p className="text-xs text-slate-400 hidden sm:block">
              Inertial Human Activity Recognition & Model Benchmarks
            </p>
          </div>
        </div>
      </div>

      <div className="flex items-center gap-3">
        {/* Backend health status badge */}
        <div className="flex items-center gap-2 bg-[#111827] border border-slate-800 px-3 py-1.5 rounded-lg text-xs">
          {offline ? (
            <div className="flex items-center gap-1.5 text-rose-400 font-medium">
              <span className="w-2 h-2 rounded-full bg-rose-500 animate-pulse" />
              <span>Backend Offline (Port 8000)</span>
            </div>
          ) : health?.status === 'healthy' ? (
            <div className="flex items-center gap-2">
              <span className="flex items-center gap-1.5 text-emerald-400 font-medium">
                <span className="w-2 h-2 rounded-full bg-emerald-500" />
                <span>API Ready</span>
              </span>
              <span className="text-slate-600">|</span>
              <span className="text-slate-400 flex items-center gap-1">
                <Cpu className="w-3.5 h-3.5 text-slate-500" />
                <span className="uppercase">{health.device}</span>
              </span>
              <span className="text-slate-600">|</span>
              <span className="text-slate-300 font-medium">
                {health.models_loaded_count}/{health.total_models} Models
              </span>
            </div>
          ) : (
            <div className="flex items-center gap-1.5 text-amber-400 font-medium">
              <AlertCircle className="w-3.5 h-3.5 text-amber-400" />
              <span>Degraded ({health?.models_loaded_count || 0}/{health?.total_models || 3})</span>
            </div>
          )}

          <button
            onClick={fetchHealth}
            disabled={isRefreshing}
            title="Refresh backend status"
            aria-label="Refresh backend status"
            className="text-slate-400 hover:text-white transition-colors ml-1 p-0.5"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isRefreshing ? 'animate-spin text-blue-400' : ''}`} />
          </button>
        </div>
      </div>
    </header>
  );
};
