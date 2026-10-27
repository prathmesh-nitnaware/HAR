import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  Radio,
  Camera,
  Cpu,
  FileSpreadsheet,
  ChevronLeft,
  ChevronRight,
  ExternalLink,
} from 'lucide-react';
import { useAnalysis } from '../context/AnalysisContext';

interface SidebarProps {
  isOpen: boolean;
  onToggle: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ isOpen, onToggle }) => {
  const { currentAnalysis } = useAnalysis();

  const navItems = [
    {
      to: '/',
      label: 'Dashboard',
      icon: LayoutDashboard,
      description: 'System health & model benchmarks',
    },
    {
      to: '/vision',
      label: 'Photo & Vision AI',
      icon: Camera,
      description: 'Camera snapshot & photo posture AI',
      badge: 'NEW',
    },
    {
      to: '/stream',
      label: 'Live Stream Studio',
      icon: Radio,
      description: '50 Hz real-time mobile & simulator',
      badge: 'LIVE',
    },
    {
      to: '/analyzer',
      label: 'Activity Analyzer',
      icon: Cpu,
      description: 'Sample benchmark & CSV inference',
    },
    {
      to: '/results',
      label: 'Results & Evaluation',
      icon: FileSpreadsheet,
      description: 'Prediction results & model confusion',
      badge: currentAnalysis ? `${currentAnalysis.n_windows}w` : undefined,
    },
  ];

  return (
    <>
      {/* Mobile overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/60 z-30 lg:hidden backdrop-blur-sm"
          onClick={onToggle}
          aria-hidden="true"
        />
      )}

      <aside
        className={`fixed lg:static top-16 bottom-0 left-0 z-40 bg-[#0b0f17] border-r border-slate-800/80 transition-all duration-200 flex flex-col justify-between ${
          isOpen ? 'w-64' : 'w-16 hidden lg:flex'
        }`}
      >
        <div className="py-4 px-2 space-y-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.to === '/'}
                title={!isOpen ? item.label : undefined}
                className={({ isActive }) =>
                  `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors group relative ${
                    isActive
                      ? 'bg-blue-600/15 text-blue-400 border border-blue-500/30'
                      : 'text-slate-400 hover:text-white hover:bg-slate-800/50 border border-transparent'
                  }`
                }
              >
                <Icon className="w-5 h-5 shrink-0" />
                {isOpen && (
                  <div className="flex-1 flex items-center justify-between overflow-hidden">
                    <span className="truncate">{item.label}</span>
                    {item.badge && (
                      <span className="text-[11px] px-1.5 py-0.5 rounded-md font-mono bg-blue-950 text-blue-300 border border-blue-800/80">
                        {item.badge}
                      </span>
                    )}
                  </div>
                )}
              </NavLink>
            );
          })}
        </div>

        {/* Footer controls & quick link */}
        <div className="p-3 border-t border-slate-800/80 space-y-2">
          {isOpen && (
            <div className="bg-[#111827] border border-slate-800 rounded-lg p-2.5">
              <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-1">
                API Docs
              </div>
              <a
                href="http://127.0.0.1:8000/docs"
                target="_blank"
                rel="noreferrer"
                className="text-xs text-blue-400 hover:text-blue-300 flex items-center justify-between"
              >
                <span>FastAPI Swagger UI</span>
                <ExternalLink className="w-3.5 h-3.5" />
              </a>
            </div>
          )}

          <button
            onClick={onToggle}
            aria-label={isOpen ? 'Collapse navigation' : 'Expand navigation'}
            className="w-full flex items-center justify-center p-2 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800/60 transition-colors"
          >
            {isOpen ? <ChevronLeft className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
          </button>
        </div>
      </aside>
    </>
  );
};
