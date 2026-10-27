import React, { useState, useEffect, useRef } from 'react';
import {
  Camera,
  Upload,
  Smartphone,
  CheckCircle2,
  AlertCircle,
  Activity,
  Radio,
  Monitor,
  RefreshCw,
  Sparkles,
  ArrowRight,
} from 'lucide-react';
import { apiService } from '../api/client';

export const MobilePage: React.FC = () => {
  const [activeMode, setActiveMode] = useState<'camera' | 'sensors'>('camera');
  const [statusMessage, setStatusMessage] = useState<string>('Connected to PC host');
  const [hostIp, setHostIp] = useState<string>('10.154.15.179');

  // Photo state
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [uploadSuccess, setUploadSuccess] = useState<boolean>(false);
  const [lastPrediction, setLastPrediction] = useState<{
    activity: string;
    confidence: number;
  } | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const cameraInputRef = useRef<HTMLInputElement | null>(null);
  const galleryInputRef = useRef<HTMLInputElement | null>(null);

  // Sensor streaming state
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [sensorValues, setSensorValues] = useState<{
    ax: number;
    ay: number;
    az: number;
    gx: number;
    gy: number;
    gz: number;
  }>({ ax: 0, ay: 0, az: 0, gx: 0, gy: 0, gz: 0 });
  const [packetsSent, setPacketsSent] = useState<number>(0);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    // Detect host from window or network info
    if (typeof window !== 'undefined' && window.location.hostname && window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
      setHostIp(window.location.hostname);
    }
    apiService.getNetworkInfo().then((info) => {
      if (info.local_ip) {
        setHostIp(info.local_ip);
      }
    }).catch(() => {
      // Keep default
    });
  }, []);

  // Handle Photo selection
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      setUploadSuccess(false);
      setLastPrediction(null);
      setErrorMessage(null);
    }
  };

  // Upload and project to PC screen
  const handleSendToPc = async () => {
    if (!selectedFile) return;
    setIsUploading(true);
    setErrorMessage(null);

    try {
      const response = await apiService.uploadRemotePhoto(selectedFile);
      setUploadSuccess(true);
      setLastPrediction({
        activity: response.predicted_activity || 'standing',
        confidence: response.confidence || 0.95,
      });
      setStatusMessage('Photo transmitted to PC display!');
    } catch (err: any) {
      setErrorMessage(err.detail || 'Failed to send photo to PC. Ensure PC is running.');
    } finally {
      setIsUploading(false);
    }
  };

  // Reset photo state
  const handleResetPhoto = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setUploadSuccess(false);
    setLastPrediction(null);
    setErrorMessage(null);
  };

  // Sensor Streaming Handler
  const toggleStreaming = async () => {
    if (isStreaming) {
      // Stop
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      setIsStreaming(false);
      return;
    }

    // Request iOS motion permissions if available
    if (
      typeof (DeviceMotionEvent as any) !== 'undefined' &&
      typeof (DeviceMotionEvent as any).requestPermission === 'function'
    ) {
      try {
        const perm = await (DeviceMotionEvent as any).requestPermission();
        if (perm !== 'granted') {
          setErrorMessage('Motion sensor permission was denied by device.');
          return;
        }
      } catch (err) {
        console.warn('Motion permission request failed', err);
      }
    }

    // Connect WebSocket
    try {
      const wsUrl = apiService.getWebSocketUrl();
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setIsStreaming(true);
        setPacketsSent(0);
        setStatusMessage('Streaming 50 Hz motion data to PC!');
      };

      ws.onerror = (err) => {
        console.error('WebSocket streaming error', err);
        setErrorMessage('Cannot establish streaming connection to PC.');
        setIsStreaming(false);
      };

      ws.onclose = () => {
        setIsStreaming(false);
      };

      // Listen to DeviceMotionEvent
      const handleMotion = (event: DeviceMotionEvent) => {
        if (!ws || ws.readyState !== WebSocket.OPEN) return;

        const acc = (event.accelerationIncludingGravity || event.acceleration || {}) as {
          x?: number | null;
          y?: number | null;
          z?: number | null;
        };
        const rot = (event.rotationRate || {}) as {
          alpha?: number | null;
          beta?: number | null;
          gamma?: number | null;
        };

        const ax = Number(((acc.x || 0) / 9.80665).toFixed(4));
        const ay = Number(((acc.y || 0) / 9.80665).toFixed(4));
        const az = Number(((acc.z || 0) / 9.80665).toFixed(4));

        const gx = Number((((rot.alpha || 0) * Math.PI) / 180).toFixed(4));
        const gy = Number((((rot.beta || 0) * Math.PI) / 180).toFixed(4));
        const gz = Number((((rot.gamma || 0) * Math.PI) / 180).toFixed(4));

        setSensorValues({ ax, ay, az, gx, gy, gz });
        setPacketsSent((p) => p + 1);

        ws.send(
          JSON.stringify({
            action: 'push',
            source: 'remote_phone',
            broadcast: true,
            data: {
              accel_x: ax,
              accel_y: ay,
              accel_z: az,
              gyro_x: gx,
              gyro_y: gy,
              gyro_z: gz,
              timestamp: Date.now() / 1000,
            },
          })
        );
      };

      window.addEventListener('devicemotion', handleMotion);

      ws.addEventListener('close', () => {
        window.removeEventListener('devicemotion', handleMotion);
      });
    } catch (err: any) {
      setErrorMessage('Could not open sensor WebSocket: ' + err.message);
    }
  };

  return (
    <div className="min-h-screen bg-[#070b14] text-slate-100 flex flex-col font-sans p-4 max-w-md mx-auto selection:bg-blue-600">
      {/* Mobile Top Header */}
      <header className="flex items-center justify-between pb-4 border-b border-slate-800/80 mb-4">
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-xl bg-gradient-to-tr from-blue-600 to-indigo-600 text-white shadow-lg shadow-blue-500/20">
            <Smartphone className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-base font-bold text-white tracking-tight flex items-center gap-1.5">
              <span>MotionSense AI</span>
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-500/20 text-blue-400 font-mono">REMOTE</span>
            </h1>
            <p className="text-[11px] text-slate-400">Wireless Phone Companion</p>
          </div>
        </div>

        {/* Connection Indicator */}
        <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs font-medium">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          <span className="text-[11px] font-mono">{hostIp}</span>
        </div>
      </header>

      {/* Target Notice: Everything appears on PC */}
      <div className="p-3 bg-blue-950/30 border border-blue-500/30 rounded-xl mb-4 flex items-start gap-2.5 text-xs text-blue-200">
        <Monitor className="w-4 h-4 text-blue-400 shrink-0 mt-0.5" />
        <div className="leading-relaxed">
          <span className="font-semibold text-white">Live PC Display Active: </span>
          Photos taken and sensor data streamed from this phone will project instantly onto your PC monitor.
        </div>
      </div>

      {/* Mode Switcher Tabs */}
      <div className="flex p-1 bg-slate-900 rounded-xl border border-slate-800 mb-4">
        <button
          onClick={() => setActiveMode('camera')}
          className={`flex-1 flex items-center justify-center gap-2 py-2.5 px-3 rounded-lg text-xs font-semibold transition-all ${
            activeMode === 'camera'
              ? 'bg-blue-600 text-white shadow-md shadow-blue-500/25'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          <Camera className="w-4 h-4" />
          <span>Camera Scanner</span>
        </button>

        <button
          onClick={() => setActiveMode('sensors')}
          className={`flex-1 flex items-center justify-center gap-2 py-2.5 px-3 rounded-lg text-xs font-semibold transition-all ${
            activeMode === 'sensors'
              ? 'bg-emerald-600 text-white shadow-md shadow-emerald-500/25'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          <Radio className="w-4 h-4" />
          <span>50 Hz IMU Stream</span>
        </button>
      </div>

      {/* Error Alert */}
      {errorMessage && (
        <div className="p-3 bg-rose-950/40 border border-rose-800/60 rounded-xl text-xs text-rose-300 flex items-start gap-2 mb-4 animate-shake">
          <AlertCircle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
          <div className="flex-1">{errorMessage}</div>
        </div>
      )}

      {/* MODE 1: CAMERA SCANNER */}
      {activeMode === 'camera' && (
        <div className="flex-1 flex flex-col space-y-4">
          {/* Hidden File Inputs */}
          <input
            type="file"
            ref={cameraInputRef}
            accept="image/*"
            capture="environment"
            className="hidden"
            onChange={handleFileChange}
          />
          <input
            type="file"
            ref={galleryInputRef}
            accept="image/*"
            className="hidden"
            onChange={handleFileChange}
          />

          {/* Action Trigger Buttons */}
          {!previewUrl && (
            <div className="space-y-3">
              <button
                onClick={() => cameraInputRef.current?.click()}
                className="w-full py-4 px-5 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-semibold rounded-2xl shadow-xl shadow-blue-600/20 flex items-center justify-center gap-3 transition-transform active:scale-[0.98]"
              >
                <div className="p-2 rounded-xl bg-white/20">
                  <Camera className="w-6 h-6 text-white" />
                </div>
                <div className="text-left">
                  <div className="text-base font-bold">Take Photo (Phone Camera)</div>
                  <div className="text-xs text-blue-200">Opens native camera to capture human pose</div>
                </div>
              </button>

              <button
                onClick={() => galleryInputRef.current?.click()}
                className="w-full py-3.5 px-5 bg-slate-900 hover:bg-slate-800 text-slate-200 font-medium rounded-2xl border border-slate-800 flex items-center justify-center gap-3 transition-colors active:scale-[0.98]"
              >
                <Upload className="w-5 h-5 text-slate-400" />
                <span className="text-sm">Choose from Phone Gallery</span>
              </button>
            </div>
          )}

          {/* Image Preview & Transmit Action */}
          {previewUrl && (
            <div className="space-y-4 flex-1 flex flex-col">
              <div className="relative rounded-2xl overflow-hidden border border-slate-700/80 bg-black/60 aspect-[3/4] max-h-72 flex items-center justify-center shadow-2xl">
                <img
                  src={previewUrl}
                  alt="Selected"
                  className="w-full h-full object-contain"
                />

                {uploadSuccess && (
                  <div className="absolute inset-0 bg-emerald-950/70 backdrop-blur-sm flex flex-col items-center justify-center p-4 text-center animate-fade-in">
                    <CheckCircle2 className="w-12 h-12 text-emerald-400 mb-2 animate-bounce" />
                    <div className="text-lg font-bold text-white">Projected onto PC!</div>
                    <div className="text-xs text-emerald-200 mt-1 max-w-[220px]">
                      Look at your PC monitor screen to view the 13-point skeleton overlay and angles.
                    </div>
                    {lastPrediction && (
                      <div className="mt-3 px-3 py-1.5 bg-emerald-500/20 border border-emerald-500/40 rounded-xl text-emerald-300 font-mono text-sm font-bold">
                        {lastPrediction.activity.toUpperCase()} ({(lastPrediction.confidence * 100).toFixed(1)}%)
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Upload Action / Reset */}
              {!uploadSuccess ? (
                <div className="flex gap-2">
                  <button
                    onClick={handleResetPhoto}
                    className="py-3 px-4 bg-slate-800 hover:bg-slate-700 text-slate-300 font-medium rounded-xl text-xs transition-colors"
                  >
                    Retake
                  </button>
                  <button
                    onClick={handleSendToPc}
                    disabled={isUploading}
                    className="flex-1 py-3 px-5 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white font-bold rounded-xl shadow-lg shadow-emerald-600/25 flex items-center justify-center gap-2 transition-transform active:scale-[0.98] disabled:opacity-50"
                  >
                    {isUploading ? (
                      <>
                        <RefreshCw className="w-4 h-4 animate-spin" />
                        <span>Transmitting to PC...</span>
                      </>
                    ) : (
                      <>
                        <Monitor className="w-4 h-4" />
                        <span>Send to PC Screen</span>
                        <ArrowRight className="w-4 h-4" />
                      </>
                    )}
                  </button>
                </div>
              ) : (
                <button
                  onClick={handleResetPhoto}
                  className="w-full py-3 px-4 bg-blue-600 hover:bg-blue-500 text-white font-bold rounded-xl text-sm transition-colors flex items-center justify-center gap-2"
                >
                  <Camera className="w-4 h-4" />
                  <span>Take Another Photo</span>
                </button>
              )}
            </div>
          )}

          {/* Quick Help Card */}
          <div className="mt-auto p-3.5 bg-slate-900/60 rounded-xl border border-slate-800/80 text-xs text-slate-400 space-y-1.5">
            <div className="font-semibold text-slate-300 flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-blue-400" />
              <span>Posture Recognition Tips:</span>
            </div>
            <p className="text-[11px] leading-relaxed">
              Capture full body from head to feet. Ideal distance: 2–3 meters. Standing, Walking, Sitting, Laying, or Fall Collapses are recognized in real time.
            </p>
          </div>
        </div>
      )}

      {/* MODE 2: 50 Hz IMU SENSORS */}
      {activeMode === 'sensors' && (
        <div className="flex-1 flex flex-col space-y-4">
          <div className="p-4 bg-slate-900/60 border border-slate-800 rounded-2xl space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-300">50 Hz Sensor Pipeline</span>
              <span
                className={`px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider ${
                  isStreaming ? 'bg-emerald-500/20 text-emerald-400 animate-pulse' : 'bg-slate-800 text-slate-400'
                }`}
              >
                {isStreaming ? 'Active Live' : 'Standby'}
              </span>
            </div>

            {/* Live Readings HUD */}
            <div className="grid grid-cols-3 gap-2 text-center">
              <div className="p-2.5 bg-slate-950/80 rounded-xl border border-slate-800/80">
                <div className="text-[10px] text-slate-500 uppercase font-mono">Accel X</div>
                <div className="text-sm font-mono font-bold text-blue-400">{sensorValues.ax.toFixed(2)}g</div>
              </div>
              <div className="p-2.5 bg-slate-950/80 rounded-xl border border-slate-800/80">
                <div className="text-[10px] text-slate-500 uppercase font-mono">Accel Y</div>
                <div className="text-sm font-mono font-bold text-emerald-400">{sensorValues.ay.toFixed(2)}g</div>
              </div>
              <div className="p-2.5 bg-slate-950/80 rounded-xl border border-slate-800/80">
                <div className="text-[10px] text-slate-500 uppercase font-mono">Accel Z</div>
                <div className="text-sm font-mono font-bold text-purple-400">{sensorValues.az.toFixed(2)}g</div>
              </div>
            </div>

            <div className="text-center text-xs text-slate-400 font-mono">
              Packets Transmitted to PC: <span className="text-emerald-400 font-bold">{packetsSent}</span>
            </div>
          </div>

          {/* Big Stream Toggle Button */}
          <button
            onClick={toggleStreaming}
            className={`w-full py-4 px-6 rounded-2xl font-bold text-base flex items-center justify-center gap-3 transition-transform active:scale-[0.98] shadow-xl ${
              isStreaming
                ? 'bg-rose-600 hover:bg-rose-500 text-white shadow-rose-600/25'
                : 'bg-emerald-600 hover:bg-emerald-500 text-white shadow-emerald-600/25'
            }`}
          >
            {isStreaming ? (
              <>
                <Radio className="w-5 h-5 animate-pulse" />
                <span>Stop Sensor Streaming</span>
              </>
            ) : (
              <>
                <Activity className="w-5 h-5" />
                <span>Start 50 Hz Streaming to PC</span>
              </>
            )}
          </button>

          <div className="p-3 bg-amber-500/10 border border-amber-500/20 rounded-xl text-xs text-amber-200 flex items-start gap-2">
            <Monitor className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
            <div className="leading-relaxed">
              Open the <strong>Live Stream</strong> page on your PC monitor to view the rolling 60 FPS oscilloscope and emergency fall detection.
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
