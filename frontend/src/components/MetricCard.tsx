import React from 'react';

interface MetricCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  badge?: {
    text: string;
    variant?: 'success' | 'warning' | 'info' | 'default';
  };
  icon?: React.ReactNode;
  loading?: boolean;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  subtitle,
  badge,
  icon,
  loading,
}) => {
  if (loading) {
    return (
      <div className="bg-[#111827] border border-slate-800 rounded-xl p-5 animate-pulse">
        <div className="h-4 bg-slate-800 rounded w-1/3 mb-3"></div>
        <div className="h-8 bg-slate-800 rounded w-2/3 mb-2"></div>
        <div className="h-3 bg-slate-800 rounded w-1/2"></div>
      </div>
    );
  }

  const badgeVariants = {
    success: 'bg-emerald-950/80 text-emerald-400 border-emerald-800/60',
    warning: 'bg-amber-950/80 text-amber-400 border-amber-800/60',
    info: 'bg-blue-950/80 text-blue-400 border-blue-800/60',
    default: 'bg-slate-800 text-slate-300 border-slate-700',
  };

  return (
    <div className="bg-[#111827] border border-slate-800 hover:border-slate-700 transition-colors rounded-xl p-5 shadow-sm">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
          {title}
        </span>
        {icon && <div className="text-slate-500">{icon}</div>}
      </div>

      <div className="flex items-baseline gap-2.5">
        <span className="text-2xl font-bold tracking-tight text-white">{value}</span>
        {badge && (
          <span
            className={`text-xs px-2 py-0.5 rounded-full border font-medium ${
              badgeVariants[badge.variant || 'default']
            }`}
          >
            {badge.text}
          </span>
        )}
      </div>

      {subtitle && <p className="text-xs text-slate-400 mt-1.5">{subtitle}</p>}
    </div>
  );
};
