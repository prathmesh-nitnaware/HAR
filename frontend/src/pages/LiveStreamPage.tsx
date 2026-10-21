import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Activity,
  Play,
  Square,
  Smartphone,
  Cpu,
  Gauge,
  Compass,
  CheckCircle2,
  AlertCircle,
  Wifi,
  WifiOff,
  Sparkles,
  Footprints,
  ArrowUpRight,
  ArrowDownRight,
  Armchair,
  User,
  Moon,
  RotateCcw,
} from 'lucide-react';
import { apiService } from '../api/client';
import type { LivePredictionMessage, LiveTelemetry } from '../types/api';

// Activity metadata and icons
const ACTIVITY_CONFIG: Record<
  string,
  { label: string; icon: React.FC<{ className?: string }>; color: string; bg: string; border: string }
> = {
  walking: {
    label: 'Walking',
    icon: Footprints,
    color: 'text-emerald-400',
    bg: 'bg-emerald-500/10',
    border: 'border-emerald-500/30',
  },
  'walking-upstairs': {
    label: 'Walking Upstairs',
    icon: ArrowUpRight,
    color: 'text-cyan-400',
    bg: 'bg-cyan-500/10',
    border: 'border-cyan-500/30',
  },
  'walking-downstairs': {
    label: 'Walking Downstairs',
    icon: ArrowDownRight,
    color: 'text-sky-400',
    bg: 'bg-sky-500/10',
    border: 'border-sky-500/30',
  },
  sitting: {
    label: 'Sitting',
    icon: Armchair,
    color: 'text-amber-400',
    bg: 'bg-amber-500/10',
    border: 'border-amber-500/30',
  },
  standing: {
    label: 'Standing',
    icon: User,
    color: 'text-purple-400',
    bg: 'bg-purple-500/10',
    border: 'border-purple-500/30',
  },
  laying: {
    label: 'Laying',
    icon: Moon,
    color: 'text-indigo-400',
    bg: 'bg-indigo-500/10',
    border: 'border-indigo-500/30',
  },
};

const DEFAULT_ACTIVITIES = [
  'walking',
  'walking-upstairs',
  'walking-downstairs',
  'sitting',
  'standing',
  'laying',
];

export const LiveStreamPage: React.FC = () => {
  // Connection state
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [wsError, setWsError] = useState<string | null>(null);
  const [activeModel, setActiveModel] = useState<string>('convlstm');
  const [availableModels, setAvailableModels] = useState<string[]>([
    'convlstm',
    'cnn1d',
    'gru',
    'lstm',
  ]);

  // Mode: 'simulate' | 'phone'
  const [streamMode, setStreamMode] = useState<'simulate' | 'phone'>('simulate');
  const [isStreaming, setIsStreaming] = useState<boolean>(false);

  // Simulation controls
  const [simActivity, setSimActivity] = useState<string>('walking');
  const [simulationSamples, setSimulationSamples] = useState<
    Record<string, { name: string; activity: string; readings: any[] }>
  >({});
  const [simSpeed, setSimSpeed] = useState<number>(1);
  const simIntervalRef = useRef<number | null>(null);
  const simIndexRef = useRef<number>(0);

  // Phone sensor state
  const [isPhoneSensorActive, setIsPhoneSensorActive] = useState<boolean>(false);
  const [sensorPermissionGranted, setSensorPermissionGranted] = useState<boolean | null>(null);
  const [phoneDeviceSupported, setPhoneDeviceSupported] = useState<boolean>(true);

  // Live telemetry & prediction
  const [latestPrediction, setLatestPrediction] = useState<LivePredictionMessage | null>(null);
  const [latestTelemetry, setLatestTelemetry] = useState<LiveTelemetry>({
    magnitude_g: 0.98,
    tilt_angle_deg: 0,
    state: 'static',
    sample_count: 0,
  });
  const [windowCount, setWindowCount] = useState<number>(0);

  // WebSocket ref
  const wsRef = useRef<WebSocket | null>(null);

  // Oscilloscope Canvas ref & rolling waveform buffer
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const waveformBufferRef = useRef<Array<{ ax: number; ay: number; az: number; gx: number; gy: number; gz: number }>>([]);

  // Connect to WebSocket
  const connectWebSocket = useCallback(() => {
    try {
      const wsUrl = apiService.getWebSocketUrl();
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        setIsConnected(true);
        setWsError(null);
        // Configure initial model
        ws.send(JSON.stringify({ action: 'configure', model: activeModel, stride: 16 }));
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'connected') {
            if (data.available_models) setAvailableModels(data.available_models);
            if (data.active_model) setActiveModel(data.active_model);
          } else if (data.type === 'prediction') {
            setLatestPrediction(data);
            if (data.telemetry) setLatestTelemetry(data.telemetry);
            setWindowCount(data.window_count || ((prev) => prev + 1));
          } else if (data.type === 'telemetry') {
            if (data.telemetry) setLatestTelemetry(data.telemetry);
          }
        } catch {
          // ignore parsing error
        }
      };

      ws.onerror = () => {
        setWsError('WebSocket connection error. Is the FastAPI backend running?');
        setIsConnected(false);
      };

      ws.onclose = () => {
        setIsConnected(false);
      };

      wsRef.current = ws;
    } catch (err: any) {
      setWsError(err.message || 'Failed to initialize WebSocket');
      setIsConnected(false);
    }
  }, [activeModel]);

  // Initialize on mount
  useEffect(() => {
    connectWebSocket();

    // Load simulation dataset samples
    apiService
      .getSimulationSamples()
      .then((samples) => {
        setSimulationSamples(samples);
      })
      .catch(() => {
        // Fallback: will generate synthetic waveforms
      });

    // Check device motion support
    if (typeof window !== 'undefined' && !('DeviceMotionEvent' in window)) {
      setPhoneDeviceSupported(false);
    }

    return () => {
      if (wsRef.current) wsRef.current.close();
      if (simIntervalRef.current) clearInterval(simIntervalRef.current);
    };
  }, [connectWebSocket]);

  // Handle model switch
  const handleModelChange = (newModel: string) => {
    setActiveModel(newModel);
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: 'configure', model: newModel, stride: 16 }));
    }
  };

  // Push reading to buffer & oscilloscope
  const pushReading = useCallback((reading: {
    ax: number;
    ay: number;
    az: number;
    gx: number;
    gy: number;
    gz: number;
    [key: string]: any;
  }) => {
    // Add to rolling oscilloscope buffer
    const buf = waveformBufferRef.current;
    buf.push(reading);
    if (buf.length > 200) buf.shift();

    // Send to WebSocket
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({
          action: 'push',
          data: {
            accel_x: reading.ax,
            accel_y: reading.ay,
            accel_z: reading.az,
            gyro_x: reading.gx,
            gyro_y: reading.gy,
            gyro_z: reading.gz,
            ...reading,
          },
        })
      );
    }
  }, []);

  // Oscilloscope Animation Frame
  useEffect(() => {
    let animId: number;

    const renderScope = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const width = canvas.width;
      const height = canvas.height;

      // Dark background
      ctx.fillStyle = '#0b0f17';
      ctx.fillRect(0, 0, width, height);

      // Grid lines
      ctx.strokeStyle = '#1e293b';
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let x = 0; x < width; x += 40) {
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
      }
      for (let y = 0; y < height; y += 30) {
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
      }
      ctx.stroke();

      // Zero-lines (center for Gyro, center for Accel)
      const centerY = height / 2;
      ctx.strokeStyle = '#334155';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(0, centerY);
      ctx.lineTo(width, centerY);
      ctx.stroke();
      ctx.setLineDash([]);

      const buf = waveformBufferRef.current;
      if (buf.length < 2) {
        animId = requestAnimationFrame(renderScope);
        return;
      }

      // Draw 3 Accelerometer Channels (Ax: Cyan, Ay: Emerald, Az: Rose)
      const stepX = width / Math.max(1, buf.length - 1);
      const scaleY = height / 5; // ~2.5g amplitude fits nicely

      const drawSignal = (
        getValue: (r: { ax: number; ay: number; az: number; gx: number; gy: number; gz: number }) => number,
        color: string,
        lineW: number
      ) => {
        ctx.strokeStyle = color;
        ctx.lineWidth = lineW;
        ctx.beginPath();
        for (let i = 0; i < buf.length; i++) {
          const x = i * stepX;
          const val = getValue(buf[i]);
          const y = centerY - val * scaleY;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
      };

      drawSignal((r) => r.ax, '#38bdf8', 1.8); // Ax: Sky Blue
      drawSignal((r) => r.ay, '#34d399', 1.8); // Ay: Emerald
      drawSignal((r) => r.az, '#f472b6', 1.8); // Az: Rose

      animId = requestAnimationFrame(renderScope);
    };

    animId = requestAnimationFrame(renderScope);
    return () => cancelAnimationFrame(animId);
  }, []);

  // Continuous Simulation Runner (50 Hz interval = 20ms)
  const startSimulation = () => {
    setIsStreaming(true);
    if (simIntervalRef.current) clearInterval(simIntervalRef.current);

    // Check if we have dataset samples for this activity
    const samplePack = simulationSamples[simActivity];
    const readings = samplePack?.readings || [];

    const intervalMs = Math.round(20 / simSpeed);

    simIntervalRef.current = window.setInterval(() => {
      let reading: any;
      if (readings.length > 0) {
        reading = readings[simIndexRef.current % readings.length];
        simIndexRef.current += 1;
        pushReading({
          ax: reading.accel_x,
          ay: reading.accel_y,
          az: reading.accel_z,
          gx: reading.gyro_x,
          gy: reading.gyro_y,
          gz: reading.gyro_z,
          ...reading,
        });
      } else {
        // Fallback synthetic generator
        const t = (simIndexRef.current * 20) / 1000;
        simIndexRef.current += 1;
        let ax = 0,
          ay = 0,
          az = 1.0,
          gx = 0,
          gy = 0,
          gz = 0;

        if (simActivity === 'walking') {
          ax = 0.4 * Math.sin(2 * Math.PI * 1.8 * t) + (Math.random() - 0.5) * 0.1;
          ay = 0.8 * Math.cos(2 * Math.PI * 1.8 * t) + (Math.random() - 0.5) * 0.15;
          az = 0.98 + 0.3 * Math.sin(4 * Math.PI * 1.8 * t);
          gx = 0.5 * Math.sin(2 * Math.PI * 1.8 * t);
        } else if (simActivity === 'walking-upstairs') {
          ax = 0.7 * Math.sin(2 * Math.PI * 2.2 * t);
          ay = 1.2 * Math.cos(2 * Math.PI * 2.2 * t);
          az = 1.0 + 0.6 * Math.abs(Math.sin(2 * Math.PI * 2.2 * t));
        } else if (simActivity === 'sitting') {
          ax = (Math.random() - 0.5) * 0.03;
          ay = 0.12 + (Math.random() - 0.5) * 0.02;
          az = 0.99 + (Math.random() - 0.5) * 0.02;
        } else if (simActivity === 'standing') {
          ax = (Math.random() - 0.5) * 0.05;
          ay = 0.88 + (Math.random() - 0.5) * 0.03;
          az = 0.35 + (Math.random() - 0.5) * 0.03;
        } else if (simActivity === 'laying') {
          ax = 0.98 + (Math.random() - 0.5) * 0.02;
          ay = (Math.random() - 0.5) * 0.03;
          az = (Math.random() - 0.5) * 0.03;
        }

        pushReading({ ax, ay, az, gx, gy, gz });
      }
    }, intervalMs);
  };

  const stopSimulation = () => {
    setIsStreaming(false);
    if (simIntervalRef.current) {
      clearInterval(simIntervalRef.current);
      simIntervalRef.current = null;
    }
  };

  // Smartphone Sensor Handler
  const startPhoneSensors = async () => {
    try {
      // iOS 13+ permission request
      if (
        typeof DeviceMotionEvent !== 'undefined' &&
        typeof (DeviceMotionEvent as any).requestPermission === 'function'
      ) {
        const permission = await (DeviceMotionEvent as any).requestPermission();
        if (permission !== 'granted') {
          setSensorPermissionGranted(false);
          return;
        }
      }

      setSensorPermissionGranted(true);
      setIsPhoneSensorActive(true);
      setIsStreaming(true);

      const motionHandler = (event: DeviceMotionEvent) => {
        const acc = event.accelerationIncludingGravity || event.acceleration;
        const gyro = event.rotationRate;

        // Convert m/s^2 to g (approximate 9.80665)
        const ax = (acc?.x || 0) / 9.80665;
        const ay = (acc?.y || 0) / 9.80665;
        const az = (acc?.z || 0) / 9.80665;

        // Degrees/sec to rad/sec
        const gx = ((gyro?.alpha || 0) * Math.PI) / 180.0;
        const gy = ((gyro?.beta || 0) * Math.PI) / 180.0;
        const gz = ((gyro?.gamma || 0) * Math.PI) / 180.0;

        pushReading({ ax, ay, az, gx, gy, gz });
      };

      window.addEventListener('devicemotion', motionHandler);

      // Store cleanup on window
      (window as any).__motionHandler = motionHandler;
    } catch (err: any) {
      setSensorPermissionGranted(false);
      setWsError(err.message || 'Failed to activate phone sensors');
    }
  };

  const stopPhoneSensors = () => {
    setIsPhoneSensorActive(false);
    setIsStreaming(false);
    if ((window as any).__motionHandler) {
      window.removeEventListener('devicemotion', (window as any).__motionHandler);
      delete (window as any).__motionHandler;
    }
  };

  const resetBuffer = () => {
    waveformBufferRef.current = [];
    setWindowCount(0);
    setLatestPrediction(null);
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: 'reset' }));
    }
  };

  const currentActivityConfig =
    latestPrediction && ACTIVITY_CONFIG[latestPrediction.predicted_activity]
      ? ACTIVITY_CONFIG[latestPrediction.predicted_activity]
      : {
          label: 'Awaiting Stream...',
          icon: Activity,
          color: 'text-slate-400',
          bg: 'bg-slate-800/30',
          border: 'border-slate-700/50',
        };

  const CurrentIcon = currentActivityConfig.icon;

  return (
    <div className="space-y-6 pb-12">
      {/* Top Header & Connection Bar */}
      <div className="bg-[#111827] border border-slate-800/90 rounded-xl p-5 shadow-lg">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-3">
              <div className="p-2.5 rounded-lg bg-blue-600/10 border border-blue-500/30 text-blue-400">
                <Activity className="w-6 h-6 animate-pulse" />
              </div>
              <div>
                <h1 className="text-xl sm:text-2xl font-bold tracking-tight text-white flex items-center gap-2">
                  Live Sensor Streaming Studio
                  <span className="text-xs px-2.5 py-0.5 rounded-full font-semibold uppercase tracking-wider bg-gradient-to-r from-blue-600 to-indigo-600 text-white">
                    50 Hz Real-Time
                  </span>
                </h1>
                <p className="text-xs sm:text-sm text-slate-400 mt-0.5">
                  Stream continuous accelerometer & gyroscope signals from your mobile phone or simulated waveforms.
                </p>
              </div>
            </div>
          </div>

          {/* Right Controls: Model Picker & Connection Status */}
          <div className="flex flex-wrap items-center gap-3">
            {/* Connection badge */}
            <div
              className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border text-xs font-medium ${
                isConnected
                  ? 'bg-emerald-950/60 border-emerald-700/50 text-emerald-300'
                  : 'bg-red-950/60 border-red-700/50 text-red-300'
              }`}
            >
              {isConnected ? (
                <>
                  <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping" />
                  <Wifi className="w-3.5 h-3.5" />
                  <span>WebSocket Connected</span>
                </>
              ) : (
                <>
                  <WifiOff className="w-3.5 h-3.5" />
                  <span>Disconnected</span>
                  <button
                    onClick={connectWebSocket}
                    className="underline ml-1 hover:text-white"
                  >
                    Retry
                  </button>
                </>
              )}
            </div>

            {/* Model Selector */}
            <div className="flex items-center gap-2 bg-[#090d16] border border-slate-800 rounded-lg px-3 py-1.5">
              <Cpu className="w-4 h-4 text-blue-400" />
              <label htmlFor="model-select" className="text-xs text-slate-400">Model:</label>
              <select
                id="model-select"
                value={activeModel}
                onChange={(e) => handleModelChange(e.target.value)}
                className="bg-transparent text-xs font-semibold text-white focus:outline-none cursor-pointer"
              >
                {availableModels.map((m) => (
                  <option key={m} value={m} className="bg-[#111827] text-white">
                    {m === 'convlstm'
                      ? 'DeepConvLSTM (Advanced)'
                      : m.toUpperCase()}
                  </option>
                ))}
              </select>
            </div>

            {/* Reset button */}
            <button
              onClick={resetBuffer}
              title="Reset streaming buffer"
              className="p-2 rounded-lg bg-slate-800/80 hover:bg-slate-700 text-slate-300 transition-colors"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
          </div>
        </div>

        {wsError && (
          <div className="mt-4 p-3 bg-red-950/40 border border-red-800/60 rounded-lg flex items-center gap-2 text-xs text-red-300">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{wsError}</span>
          </div>
        )}
      </div>

      {/* Main Grid: Stream Source & Live HUD */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Stream Controls & Oscilloscope (7 cols) */}
        <div className="lg:col-span-7 space-y-6">
          {/* Stream Source Selector Tabs */}
          <div className="bg-[#111827] border border-slate-800 rounded-xl p-4">
            <div className="flex border-b border-slate-800 pb-3 gap-2">
              <button
                onClick={() => {
                  stopPhoneSensors();
                  setStreamMode('simulate');
                }}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold transition-all ${
                  streamMode === 'simulate'
                    ? 'bg-blue-600 text-white shadow-sm'
                    : 'text-slate-400 hover:text-white hover:bg-slate-800/50'
                }`}
              >
                <Sparkles className="w-4 h-4" />
                Desktop Simulation Streamer
              </button>

              <button
                onClick={() => {
                  stopSimulation();
                  setStreamMode('phone');
                }}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold transition-all ${
                  streamMode === 'phone'
                    ? 'bg-blue-600 text-white shadow-sm'
                    : 'text-slate-400 hover:text-white hover:bg-slate-800/50'
                }`}
              >
                <Smartphone className="w-4 h-4" />
                Live Mobile Phone Sensor
              </button>
            </div>

            {/* Mode A: Desktop Simulation Streamer */}
            {streamMode === 'simulate' && (
              <div className="mt-4 space-y-4">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <label className="text-xs font-semibold text-slate-400 block mb-1">
                      Select Target Activity Pattern:
                    </label>
                    <div className="flex flex-wrap gap-1.5">
                      {DEFAULT_ACTIVITIES.map((act) => (
                        <button
                          key={act}
                          onClick={() => {
                            setSimActivity(act);
                            if (isStreaming) {
                              stopSimulation();
                              setTimeout(startSimulation, 50);
                            }
                          }}
                          className={`text-xs px-2.5 py-1.5 rounded-md font-medium capitalize transition-colors ${
                            simActivity === act
                              ? 'bg-blue-600 text-white'
                              : 'bg-slate-800/80 text-slate-300 hover:bg-slate-700'
                          }`}
                        >
                          {act.replace('-', ' ')}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Play / Stop Buttons */}
                  <div className="flex items-center gap-2">
                    {!isStreaming ? (
                      <button
                        onClick={startSimulation}
                        disabled={!isConnected}
                        className="flex items-center gap-2 px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white text-xs font-bold transition-all shadow-md"
                      >
                        <Play className="w-4 h-4 fill-current" />
                        Start 50 Hz Stream
                      </button>
                    ) : (
                      <button
                        onClick={stopSimulation}
                        className="flex items-center gap-2 px-4 py-2 rounded-lg bg-rose-600 hover:bg-rose-500 text-white text-xs font-bold transition-all shadow-md"
                      >
                        <Square className="w-4 h-4 fill-current" />
                        Stop Stream
                      </button>
                    )}
                  </div>
                </div>

                <div className="flex items-center gap-4 text-xs text-slate-400 pt-2 border-t border-slate-800/80">
                  <div className="flex items-center gap-2">
                    <span>Speed:</span>
                    {[0.5, 1, 2].map((s) => (
                      <button
                        key={s}
                        onClick={() => setSimSpeed(s)}
                        className={`px-2 py-0.5 rounded text-[11px] font-mono ${
                          simSpeed === s
                            ? 'bg-blue-600 text-white'
                            : 'bg-slate-800 text-slate-400 hover:text-white'
                        }`}
                      >
                        {s}x
                      </button>
                    ))}
                  </div>
                  <div className="text-[11px] text-slate-500">
                    Cycles UCI HAR test split sequences with 50 Hz timestamp alignment.
                  </div>
                </div>
              </div>
            )}

            {/* Mode B: Live Mobile Phone Sensor */}
            {streamMode === 'phone' && (
              <div className="mt-4 space-y-4">
                <div className="bg-[#090d16] border border-slate-800 rounded-lg p-4">
                  <div className="flex items-start gap-3">
                    <div className="p-2 rounded-lg bg-indigo-500/10 text-indigo-400 shrink-0">
                      <Smartphone className="w-5 h-5" />
                    </div>
                    <div className="space-y-1">
                      <h4 className="text-sm font-semibold text-white">
                        Connect Your Physical Phone Over Local Wi-Fi
                      </h4>
                      <p className="text-xs text-slate-400 leading-relaxed">
                        Open this web app directly on your smartphone’s Safari or Chrome browser (on the same Wi-Fi
                        network). Click the button below to grant sensor access, then place the phone in your pocket
                        and walk, sit, or climb stairs!
                      </p>
                    </div>
                  </div>

                  <div className="mt-4 pt-3 border-t border-slate-800 flex flex-wrap items-center justify-between gap-3">
                    {!isPhoneSensorActive ? (
                      <button
                        onClick={startPhoneSensors}
                        disabled={!isConnected}
                        className="flex items-center gap-2 px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white text-xs font-bold transition-all shadow-md"
                      >
                        <Play className="w-4 h-4 fill-current" />
                        Enable Phone Motion Sensors
                      </button>
                    ) : (
                      <button
                        onClick={stopPhoneSensors}
                        className="flex items-center gap-2 px-4 py-2 rounded-lg bg-rose-600 hover:bg-rose-500 text-white text-xs font-bold transition-all shadow-md"
                      >
                        <Square className="w-4 h-4 fill-current" />
                        Stop Phone Sensors
                      </button>
                    )}

                    {sensorPermissionGranted === false && (
                      <span className="text-xs text-amber-400">
                        Sensor permission denied or not supported on this browser.
                      </span>
                    )}
                    {!phoneDeviceSupported && (
                      <span className="text-xs text-slate-500">
                        (DeviceMotionEvent not detected on desktop; use Mobile mode on your phone)
                      </span>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Real-Time Oscilloscope Waveform */}
          <div className="bg-[#111827] border border-slate-800 rounded-xl p-4 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Gauge className="w-4 h-4 text-cyan-400" />
                <h3 className="text-sm font-bold text-white">
                  Real-Time IMU Oscilloscope (50 Hz Rolling Buffer)
                </h3>
              </div>
              <div className="flex items-center gap-3 text-[11px] font-mono">
                <span className="flex items-center gap-1 text-sky-400">
                  <span className="w-2 h-2 rounded-full bg-sky-400" /> Ax
                </span>
                <span className="flex items-center gap-1 text-emerald-400">
                  <span className="w-2 h-2 rounded-full bg-emerald-400" /> Ay
                </span>
                <span className="flex items-center gap-1 text-pink-400">
                  <span className="w-2 h-2 rounded-full bg-pink-400" /> Az
                </span>
              </div>
            </div>

            <div className="relative border border-slate-800 rounded-lg overflow-hidden bg-[#0b0f17]">
              <canvas
                ref={canvasRef}
                width={700}
                height={220}
                className="w-full h-52 block"
              />
              <div className="absolute top-2 left-2 px-2 py-0.5 rounded bg-black/60 border border-slate-800 text-[10px] font-mono text-slate-400">
                Range: ±2.5g
              </div>
              <div className="absolute bottom-2 right-2 px-2 py-0.5 rounded bg-black/60 border border-slate-800 text-[10px] font-mono text-slate-400">
                Stride: 16 samples (~320ms)
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Live Classification HUD & Probabilities (5 cols) */}
        <div className="lg:col-span-5 space-y-6">
          {/* Hero Classified Activity Card */}
          <div
            className={`border rounded-xl p-6 transition-all shadow-xl relative overflow-hidden ${currentActivityConfig.bg} ${currentActivityConfig.border}`}
          >
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold uppercase tracking-wider text-slate-400">
                Live Activity Classification
              </span>
              <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-slate-900/80 border border-slate-700/60 text-slate-300">
                Window #{windowCount}
              </span>
            </div>

            <div className="mt-4 flex items-center gap-4">
              <div
                className={`p-4 rounded-2xl border shadow-inner ${currentActivityConfig.bg} ${currentActivityConfig.border} ${currentActivityConfig.color}`}
              >
                <CurrentIcon className="w-10 h-10" />
              </div>
              <div>
                <div className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
                  {latestPrediction ? latestPrediction.predicted_label : 'Waiting for Data'}
                </div>
                <div className="flex items-center gap-2 mt-1">
                  <span className="text-xs font-semibold text-slate-300">
                    Confidence:{' '}
                    <span className="text-white font-mono font-bold">
                      {latestPrediction ? `${(latestPrediction.confidence * 100).toFixed(1)}%` : '--'}
                    </span>
                  </span>
                  {latestPrediction && (
                    <span className="text-[10px] px-1.5 py-0.5 rounded font-mono bg-blue-900/60 text-blue-300 border border-blue-700/60">
                      {latestPrediction.latency_ms} ms
                    </span>
                  )}
                </div>
              </div>
            </div>

            {/* Animated Confidence Bar */}
            <div className="mt-5 space-y-1.5">
              <div className="h-2 w-full bg-slate-800/80 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-blue-500 to-indigo-500 rounded-full transition-all duration-200"
                  style={{
                    width: latestPrediction ? `${latestPrediction.confidence * 100}%` : '0%',
                  }}
                />
              </div>
            </div>
          </div>

          {/* Real-Time Probability Distribution */}
          <div className="bg-[#111827] border border-slate-800 rounded-xl p-4 space-y-3">
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-2">
              <CheckCircle2 className="w-3.5 h-3.5 text-blue-400" />
              Real-Time Class Probabilities
            </h4>

            <div className="space-y-2 pt-1">
              {DEFAULT_ACTIVITIES.map((act) => {
                const prob = latestPrediction?.probabilities?.[act] ?? 0;
                const isWinner = latestPrediction?.predicted_activity === act;
                const cfg = ACTIVITY_CONFIG[act];
                const Icon = cfg.icon;

                return (
                  <div key={act} className="space-y-1">
                    <div className="flex items-center justify-between text-xs">
                      <span className="flex items-center gap-2 text-slate-300 font-medium">
                        <Icon className={`w-3.5 h-3.5 ${isWinner ? cfg.color : 'text-slate-500'}`} />
                        {cfg.label}
                      </span>
                      <span className="font-mono text-[11px] font-bold text-slate-200">
                        {(prob * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all duration-150 ${
                          isWinner ? 'bg-blue-500' : 'bg-slate-700'
                        }`}
                        style={{ width: `${prob * 100}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Biomechanical Telemetry Gauges */}
          <div className="bg-[#111827] border border-slate-800 rounded-xl p-4 space-y-3">
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-2">
              <Compass className="w-3.5 h-3.5 text-cyan-400" />
              Biomechanical Telemetry
            </h4>

            <div className="grid grid-cols-3 gap-3">
              <div className="bg-[#090d16] border border-slate-800/80 rounded-lg p-3 text-center">
                <div className="text-[10px] text-slate-400 uppercase font-semibold">Magnitude</div>
                <div className="text-lg font-mono font-extrabold text-cyan-400 mt-1">
                  {latestTelemetry.magnitude_g.toFixed(2)}
                  <span className="text-xs text-slate-500 font-normal ml-0.5">g</span>
                </div>
              </div>

              <div className="bg-[#090d16] border border-slate-800/80 rounded-lg p-3 text-center">
                <div className="text-[10px] text-slate-400 uppercase font-semibold">Tilt Angle</div>
                <div className="text-lg font-mono font-extrabold text-amber-400 mt-1">
                  {latestTelemetry.tilt_angle_deg.toFixed(1)}°
                </div>
              </div>

              <div className="bg-[#090d16] border border-slate-800/80 rounded-lg p-3 text-center">
                <div className="text-[10px] text-slate-400 uppercase font-semibold">State</div>
                <div className="mt-1">
                  <span
                    className={`text-xs px-2 py-0.5 rounded-full font-bold uppercase ${
                      latestTelemetry.state === 'dynamic'
                        ? 'bg-emerald-950 text-emerald-400 border border-emerald-800'
                        : 'bg-indigo-950 text-indigo-400 border border-indigo-800'
                    }`}
                  >
                    {latestTelemetry.state}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
