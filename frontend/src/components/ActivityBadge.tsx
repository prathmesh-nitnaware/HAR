import React from 'react';

interface ActivityBadgeProps {
  activity: string;
  label?: string;
  className?: string;
}

export const ACTIVITY_STYLES: Record<string, { bg: string; text: string; border: string; dot: string }> = {
  walking: {
    bg: 'bg-blue-950/60',
    text: 'text-blue-300',
    border: 'border-blue-800/60',
    dot: 'bg-blue-400',
  },
  'walking-upstairs': {
    bg: 'bg-indigo-950/60',
    text: 'text-indigo-300',
    border: 'border-indigo-800/60',
    dot: 'bg-indigo-400',
  },
  'walking-downstairs': {
    bg: 'bg-cyan-950/60',
    text: 'text-cyan-300',
    border: 'border-cyan-800/60',
    dot: 'bg-cyan-400',
  },
  sitting: {
    bg: 'bg-amber-950/60',
    text: 'text-amber-300',
    border: 'border-amber-800/60',
    dot: 'bg-amber-400',
  },
  standing: {
    bg: 'bg-purple-950/60',
    text: 'text-purple-300',
    border: 'border-purple-800/60',
    dot: 'bg-purple-400',
  },
  laying: {
    bg: 'bg-emerald-950/60',
    text: 'text-emerald-300',
    border: 'border-emerald-800/60',
    dot: 'bg-emerald-400',
  },
};

export const ActivityBadge: React.FC<ActivityBadgeProps> = ({ activity, label, className = '' }) => {
  const normKey = activity.toLowerCase().replace(/_/g, '-');
  const style = ACTIVITY_STYLES[normKey] || {
    bg: 'bg-slate-900',
    text: 'text-slate-300',
    border: 'border-slate-800',
    dot: 'bg-slate-400',
  };

  const displayLabel = label || activity.replace(/-/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());

  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium border ${style.bg} ${style.text} ${style.border} ${className}`}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${style.dot}`} />
      {displayLabel}
    </span>
  );
};
