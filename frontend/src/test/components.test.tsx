import React from 'react';
import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ActivityBadge } from '../components/ActivityBadge';
import { MetricCard } from '../components/MetricCard';
import { ConfusionMatrixHeatmap } from '../components/charts/ConfusionMatrixHeatmap';
import { TimelineChart } from '../components/charts/TimelineChart';

describe('Frontend UI Components', () => {
  it('renders ActivityBadge with formatted label', () => {
    render(<ActivityBadge activity="walking" label="Walking" />);
    expect(screen.getByText('Walking')).toBeInTheDocument();
  });

  it('renders MetricCard with title, value, and badge', () => {
    render(
      <MetricCard
        title="Test Accuracy"
        value="94.20%"
        subtitle="1D-CNN Top Performer"
        badge={{ text: 'Rank #1', variant: 'success' }}
      />
    );

    expect(screen.getByText('Test Accuracy')).toBeInTheDocument();
    expect(screen.getByText('94.20%')).toBeInTheDocument();
    expect(screen.getByText('Rank #1')).toBeInTheDocument();
    expect(screen.getByText('1D-CNN Top Performer')).toBeInTheDocument();
  });

  it('renders ConfusionMatrixHeatmap and toggles between Normalized and Raw Counts', () => {
    const mockCmData = {
      model: 'cnn1d',
      display_name: '1D-CNN',
      matrix: [
        [496, 0, 0, 0, 0, 0],
        [0, 456, 14, 1, 0, 0],
        [0, 0, 420, 0, 0, 0],
        [0, 3, 0, 418, 64, 6],
        [1, 0, 0, 82, 449, 0],
        [0, 0, 0, 0, 0, 537],
      ],
      matrix_normalized: [
        [1.0, 0, 0, 0, 0, 0],
        [0, 0.9682, 0.0297, 0.0021, 0, 0],
        [0, 0, 1.0, 0, 0, 0],
        [0, 0.0061, 0, 0.8513, 0.1303, 0.0122],
        [0.0019, 0, 0, 0.1541, 0.844, 0],
        [0, 0, 0, 0, 0, 1.0],
      ],
      activities: ['walking', 'walking-upstairs', 'walking-downstairs', 'sitting', 'standing', 'laying'],
      activity_labels: ['Walking', 'Upstairs', 'Downstairs', 'Sitting', 'Standing', 'Laying'],
    };

    render(<ConfusionMatrixHeatmap data={mockCmData} />);

    expect(screen.getByText('1D-CNN Confusion Matrix')).toBeInTheDocument();
    expect(screen.getAllByText('100.0%').length).toBeGreaterThan(0);

    // Toggle to raw counts
    const rawCountsBtn = screen.getByText('Raw Counts');
    fireEvent.click(rawCountsBtn);
    expect(screen.getAllByText('496').length).toBeGreaterThan(0);
  });

  it('adapts TimelineChart cleanly for single-window analysis', () => {
    const mockSingleWindow = [
      {
        window_id: 1,
        start_time_sec: 0.0,
        end_time_sec: 2.56,
        consensus_activity: 'walking',
        consensus_label: 'Walking',
        unanimous_agreement: true,
        models: {},
      },
    ];

    render(<TimelineChart windows={mockSingleWindow} durationSec={2.56} />);

    expect(screen.getByText(/Window Prediction Timeline \(Single Window\)/i)).toBeInTheDocument();
    expect(screen.getByText(/Unanimous Model Agreement/i)).toBeInTheDocument();
  });
});
