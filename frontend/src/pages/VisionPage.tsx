import React, { useState, useEffect, useRef } from 'react';
import { useDropzone } from 'react-dropzone';
import {
  Camera,
  Upload,
  Sparkles,
  Smartphone,
  AlertTriangle,
  CheckCircle2,
  AlertCircle,
  RefreshCw,
  Eye,
  Sliders,
  Footprints,
  Armchair,
  User,
  Moon,
  ChevronRight,
  Maximize2,
  Download,
  Video,
  VideoOff,
  SwitchCamera,
  Info,
} from 'lucide-react';
import { apiService } from '../api/client';
import type {
  VisionPredictionResponse,
  VisionDemoSampleItem,
} from '../types/api';
import { MobileConnectModal } from '../components/MobileConnectModal';

const ACTIVITY_METADATA: Record<
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
    label: 'Laying / Rest',
    icon: Moon,
    color: 'text-indigo-400',
    bg: 'bg-indigo-500/10',
    border: 'border-indigo-500/30',
  },
  fall: {
    label: '🚨 Fall Accident Detected',
    icon: AlertTriangle,
    color: 'text-rose-400',
    bg: 'bg-rose-500/20',
    border: 'border-rose-500/50',
  },
};

export const VisionPage: React.FC = () => {
  // Input Mode: 'camera' | 'upload' | 'demo'
  const [activeTab, setActiveTab] = useState<'camera' | 'upload' | 'demo'>('camera');

  // Mobile Connect Modal
  const [isMobileModalOpen, setIsMobileModalOpen] = useState<boolean>(false);

  // Demo samples from backend
  const [demoSamples, setDemoSamples] = useState<VisionDemoSampleItem[]>([]);
  const [selectedDemoId, setSelectedDemoId] = useState<string>('');

  // Selected or captured image
  const [imagePreviewUrl, setImagePreviewUrl] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  // Analysis result
  const [prediction, setPrediction] = useState<VisionPredictionResponse | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Overlay toggle: true = annotated skeleton, false = raw image
  const [showAnnotated, setShowAnnotated] = useState<boolean>(true);

  // Webcam stream state
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [isWebcamActive, setIsWebcamActive] = useState<boolean>(false);
  const [webcamFacingMode, setWebcamFacingMode] = useState<'user' | 'environment'>('environment');
  const [webcamError, setWebcamError] = useState<string | null>(null);

  // Native phone camera input ref
  const mobileCameraInputRef = useRef<HTMLInputElement | null>(null);

  // Load demo samples on mount
  useEffect(() => {
    apiService
      .getVisionDemoSamples()
      .then((res) => {
        setDemoSamples(res.samples);
        if (res.samples.length > 0) {
          setSelectedDemoId(res.samples[0].id);
        }
      })
      .catch(() => {
        // demo samples optional
      });
  }, []);

  // Handle webcam stream start/stop
  const startWebcam = async () => {
    setWebcamError(null);
    try {
      if (videoRef.current && videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach((track) => track.stop());
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: webcamFacingMode,
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
        setIsWebcamActive(true);
      }
    } catch (err: any) {
      setWebcamError(
        err.message || 'Unable to access camera. Please check camera permissions or use Phone Camera.'
      );
      setIsWebcamActive(false);
    }
  };

  const stopWebcam = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }
    setIsWebcamActive(false);
  };

  // Toggle front/back camera
  const toggleCameraFacing = () => {
    const nextMode = webcamFacingMode === 'user' ? 'environment' : 'user';
    setWebcamFacingMode(nextMode);
    if (isWebcamActive) {
      setTimeout(startWebcam, 100);
    }
  };

  // Stop webcam when switching tabs or unmounting
  useEffect(() => {
    if (activeTab !== 'camera' && isWebcamActive) {
      stopWebcam();
    }
    return () => {
      stopWebcam();
    };
  }, [activeTab]);

  // Capture frame from webcam and run inference
  const captureWebcamSnapshot = async () => {
    if (!videoRef.current) return;
    try {
      const video = videoRef.current;
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const dataUrl = canvas.toDataURL('image/jpeg', 0.92);

      setImagePreviewUrl(dataUrl);
      setSelectedFile(null);
      stopWebcam();

      // Analyze base64
      await runInferenceOnBase64(dataUrl);
    } catch (err: any) {
      setError(err?.message || 'Failed to capture snapshot from webcam.');
    }
  };

  // Handle native phone camera file input
  const handleMobileCameraCapture = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files && files.length > 0) {
      const file = files[0];
      setSelectedFile(file);
      const objectUrl = URL.createObjectURL(file);
      setImagePreviewUrl(objectUrl);
      runInferenceOnFile(file);
    }
  };

  // Dropzone for general image upload
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.webp'],
    },
    maxFiles: 1,
    multiple: false,
    onDrop: (acceptedFiles) => {
      if (acceptedFiles.length > 0) {
        const file = acceptedFiles[0];
        setSelectedFile(file);
        const objectUrl = URL.createObjectURL(file);
        setImagePreviewUrl(objectUrl);
        setError(null);
        runInferenceOnFile(file);
      }
    },
  });

  // Execute inference on a File
  const runInferenceOnFile = async (file: File) => {
    setIsAnalyzing(true);
    setError(null);
    try {
      const result = await apiService.predictVisionFile(file);
      setPrediction(result);
    } catch (err: any) {
      setError(err?.detail || err?.message || 'Optical pose recognition failed on uploaded file.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Execute inference on a Base64 string
  const runInferenceOnBase64 = async (b64: string) => {
    setIsAnalyzing(true);
    setError(null);
    try {
      const result = await apiService.predictVisionBase64(b64);
      setPrediction(result);
    } catch (err: any) {
      setError(err?.detail || err?.message || 'Optical pose recognition failed on image data.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Run inference on selected demo sample
  const handleRunDemoSample = async (sample: VisionDemoSampleItem) => {
    setSelectedDemoId(sample.id);
    setImagePreviewUrl(sample.image_base64);
    setSelectedFile(null);
    await runInferenceOnBase64(sample.image_base64);
  };

  const activeMeta = prediction
    ? ACTIVITY_METADATA[prediction.predicted_activity] || {
        label: prediction.predicted_label,
        icon: User,
        color: 'text-blue-400',
        bg: 'bg-blue-500/10',
        border: 'border-blue-500/30',
      }
    : null;

  const ActiveIcon = activeMeta?.icon || User;

  return (
    <div className="space-y-6 max-w-7xl mx-auto pb-12">
      {/* Page Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="px-2 py-0.5 text-[11px] font-semibold bg-blue-500/10 text-blue-400 border border-blue-500/20 rounded-md uppercase tracking-wider">
              Computer Vision & Optical HAR
            </span>
            <span className="px-2 py-0.5 text-[11px] font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 rounded-md">
              Real-Time Camera & Upload
            </span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-white flex items-center gap-2">
            <Camera className="w-8 h-8 text-blue-400" />
            Photo & Image Activity Recognition
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Capture photos from your smartphone camera or upload any picture to detect walking, sitting, standing, laying, and falls with skeletal keypoints.
          </p>
        </div>

        {/* Pair Smartphone Button */}
        <div className="flex items-center gap-3">
          <button
            onClick={() => setIsMobileModalOpen(true)}
            className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white text-sm font-semibold shadow-lg shadow-blue-500/20 transition-all cursor-pointer"
          >
            <Smartphone className="w-4 h-4" />
            <span>Connect Phone (QR)</span>
          </button>
        </div>
      </div>

      {/* Main Content Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Image Source Selection */}
        <div className="lg:col-span-5 space-y-5">
          {/* Tabs for Source Mode */}
          <div className="p-1 bg-[#111827] border border-slate-800 rounded-xl flex gap-1">
            <button
              onClick={() => setActiveTab('camera')}
              className={`flex-1 flex items-center justify-center gap-2 py-2 px-3 rounded-lg text-xs font-semibold transition-all ${
                activeTab === 'camera'
                  ? 'bg-blue-600 text-white shadow-md'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Camera className="w-4 h-4" />
              <span>Take Photo</span>
            </button>
            <button
              onClick={() => setActiveTab('upload')}
              className={`flex-1 flex items-center justify-center gap-2 py-2 px-3 rounded-lg text-xs font-semibold transition-all ${
                activeTab === 'upload'
                  ? 'bg-blue-600 text-white shadow-md'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Upload className="w-4 h-4" />
              <span>Upload Image</span>
            </button>
            <button
              onClick={() => setActiveTab('demo')}
              className={`flex-1 flex items-center justify-center gap-2 py-2 px-3 rounded-lg text-xs font-semibold transition-all ${
                activeTab === 'demo'
                  ? 'bg-blue-600 text-white shadow-md'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Sparkles className="w-4 h-4" />
              <span>Demo Presets</span>
            </button>
          </div>

          {/* TAB 1: Camera Mode */}
          {activeTab === 'camera' && (
            <div className="bg-[#111827] border border-slate-800 rounded-2xl p-5 space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="font-semibold text-white text-sm">Smartphone Camera & Webcam</h3>
                  <p className="text-xs text-slate-400">Snap a photo directly with your camera</p>
                </div>
                {isWebcamActive && (
                  <button
                    onClick={toggleCameraFacing}
                    title="Switch Camera (Front/Back)"
                    className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors"
                  >
                    <SwitchCamera className="w-4 h-4" />
                  </button>
                )}
              </div>

              {/* Native Phone Camera Input Trigger */}
              <input
                ref={mobileCameraInputRef}
                type="file"
                accept="image/*"
                capture="environment"
                onChange={handleMobileCameraCapture}
                className="hidden"
                id="native-phone-camera-input"
              />

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <button
                  onClick={() => mobileCameraInputRef.current?.click()}
                  className="flex flex-col items-center justify-center gap-2 p-4 rounded-xl bg-blue-600/15 border border-blue-500/30 hover:bg-blue-600/25 text-blue-300 transition-all text-center group cursor-pointer"
                >
                  <div className="p-3 rounded-full bg-blue-600/30 group-hover:scale-110 transition-transform">
                    <Camera className="w-6 h-6 text-blue-400" />
                  </div>
                  <span className="text-xs font-bold text-white">Click Photo (Phone Camera)</span>
                  <span className="text-[11px] text-slate-400">Opens native mobile camera</span>
                </button>

                {!isWebcamActive ? (
                  <button
                    onClick={startWebcam}
                    className="flex flex-col items-center justify-center gap-2 p-4 rounded-xl bg-slate-800/50 border border-slate-700/60 hover:bg-slate-800 text-slate-300 transition-all text-center group cursor-pointer"
                  >
                    <div className="p-3 rounded-full bg-slate-700/60 group-hover:scale-110 transition-transform">
                      <Video className="w-6 h-6 text-emerald-400" />
                    </div>
                    <span className="text-xs font-bold text-white">Open Live Viewfinder</span>
                    <span className="text-[11px] text-slate-400">Desktop / In-Browser Webcam</span>
                  </button>
                ) : (
                  <button
                    onClick={stopWebcam}
                    className="flex flex-col items-center justify-center gap-2 p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 hover:bg-rose-500/20 text-rose-300 transition-all text-center group cursor-pointer"
                  >
                    <div className="p-3 rounded-full bg-rose-500/20 group-hover:scale-110 transition-transform">
                      <VideoOff className="w-6 h-6 text-rose-400" />
                    </div>
                    <span className="text-xs font-bold text-rose-200">Close Viewfinder</span>
                    <span className="text-[11px] text-slate-400">Stop camera stream</span>
                  </button>
                )}
              </div>

              {webcamError && (
                <div className="p-3 rounded-xl bg-rose-950/40 border border-rose-800/60 text-xs text-rose-300 flex items-start gap-2">
                  <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
                  <span>{webcamError}</span>
                </div>
              )}

              {/* Live Video Viewfinder */}
              {isWebcamActive && (
                <div className="space-y-3 pt-2">
                  <div className="relative rounded-xl overflow-hidden bg-black border border-slate-700 aspect-video flex items-center justify-center">
                    <video
                      ref={videoRef}
                      autoPlay
                      playsInline
                      muted
                      className="w-full h-full object-cover"
                    />
                    <div className="absolute top-2 left-2 px-2 py-1 rounded bg-black/70 backdrop-blur-sm text-[10px] text-emerald-400 font-mono flex items-center gap-1.5 border border-emerald-500/30">
                      <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                      LIVE WEBCAM
                    </div>
                  </div>

                  <button
                    onClick={captureWebcamSnapshot}
                    disabled={isAnalyzing}
                    className="w-full py-2.5 px-4 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white font-semibold text-xs transition-colors flex items-center justify-center gap-2 shadow-lg shadow-emerald-500/20 cursor-pointer"
                  >
                    <Camera className="w-4 h-4" />
                    <span>Capture Snapshot & Analyze Activity</span>
                  </button>
                </div>
              )}

              <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800/80 text-[11px] text-slate-400 space-y-1">
                <div className="flex items-center gap-1.5 text-blue-300 font-medium">
                  <Info className="w-3.5 h-3.5" />
                  <span>Tip for Best Accuracy:</span>
                </div>
                <p>
                  Ensure the person's full body or torso and legs are visible in frame against good lighting for optimal skeletal pose extraction.
                </p>
              </div>
            </div>
          )}

          {/* TAB 2: Upload File Mode */}
          {activeTab === 'upload' && (
            <div className="bg-[#111827] border border-slate-800 rounded-2xl p-5 space-y-4">
              <div>
                <h3 className="font-semibold text-white text-sm">Upload Any Image</h3>
                <p className="text-xs text-slate-400">Supports JPG, PNG, WEBP from your computer or phone</p>
              </div>

              <div
                {...getRootProps()}
                className={`border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all ${
                  isDragActive
                    ? 'border-blue-500 bg-blue-500/10'
                    : 'border-slate-700 hover:border-slate-600 bg-slate-900/40'
                }`}
              >
                <input {...getInputProps()} />
                <div className="flex flex-col items-center gap-3">
                  <div className="p-3 rounded-full bg-blue-600/20 text-blue-400 border border-blue-500/30">
                    <Upload className="w-6 h-6" />
                  </div>
                  <div>
                    <span className="text-xs font-semibold text-white">Click or drag image file here</span>
                    <p className="text-[11px] text-slate-400 mt-1">Accepts any photo of human posture</p>
                  </div>
                  {selectedFile && (
                    <div className="mt-2 px-3 py-1 bg-slate-800 rounded-full text-xs text-blue-300 font-mono truncate max-w-[260px]">
                      {selectedFile.name}
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* TAB 3: Demo Presets Mode */}
          {activeTab === 'demo' && (
            <div className="bg-[#111827] border border-slate-800 rounded-2xl p-5 space-y-3">
              <div>
                <h3 className="font-semibold text-white text-sm">Instant Pre-Loaded Postures</h3>
                <p className="text-xs text-slate-400">Click any activity card to test inference immediately</p>
              </div>

              <div className="space-y-2 max-h-[380px] overflow-y-auto pr-1">
                {demoSamples.map((sample) => {
                  const meta = ACTIVITY_METADATA[sample.activity] || {
                    label: sample.title,
                    color: 'text-slate-300',
                    bg: 'bg-slate-800',
                    border: 'border-slate-700',
                  };
                  const isSelected = selectedDemoId === sample.id;

                  return (
                    <div
                      key={sample.id}
                      onClick={() => handleRunDemoSample(sample)}
                      className={`flex items-center gap-3 p-3 rounded-xl border transition-all cursor-pointer ${
                        isSelected
                          ? 'bg-blue-600/15 border-blue-500/50 shadow-md shadow-blue-500/10'
                          : 'bg-slate-900/50 border-slate-800/80 hover:bg-slate-800/60'
                      }`}
                    >
                      {sample.image_base64 && (
                        <img
                          src={sample.image_base64}
                          alt={sample.title}
                          className="w-12 h-12 rounded-lg object-cover bg-black border border-slate-700 shrink-0"
                        />
                      )}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-bold text-white truncate">{sample.title}</span>
                          <span
                            className={`text-[10px] font-semibold px-2 py-0.5 rounded-full border ${meta.bg} ${meta.color} ${meta.border}`}
                          >
                            {meta.label}
                          </span>
                        </div>
                        <p className="text-[11px] text-slate-400 line-clamp-1 mt-0.5">
                          {sample.description}
                        </p>
                      </div>
                      <ChevronRight className="w-4 h-4 text-slate-500 shrink-0" />
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Error Notice */}
          {error && (
            <div className="p-4 rounded-xl bg-rose-950/40 border border-rose-800 text-xs text-rose-300 flex items-start gap-2.5">
              <AlertCircle className="w-4 h-4 shrink-0 mt-0.5 text-rose-400" />
              <div>
                <strong className="font-semibold block mb-0.5">Recognition Error</strong>
                <span>{error}</span>
              </div>
            </div>
          )}
        </div>

        {/* Right Column: AI Visual Analysis Results */}
        <div className="lg:col-span-7 space-y-5">
          {/* Analysis View Card */}
          <div className="bg-[#111827] border border-slate-800 rounded-2xl p-5 space-y-5">
            {/* Header & View Switcher */}
            <div className="flex items-center justify-between flex-wrap gap-2">
              <div className="flex items-center gap-2">
                <Sliders className="w-4 h-4 text-blue-400" />
                <h3 className="font-semibold text-white text-sm">Visual Skeleton & Activity Analysis</h3>
              </div>

              {prediction && (
                <div className="flex items-center gap-1 bg-slate-950 p-1 rounded-lg border border-slate-800 text-xs">
                  <button
                    onClick={() => setShowAnnotated(true)}
                    className={`px-2.5 py-1 rounded-md transition-colors ${
                      showAnnotated ? 'bg-blue-600 text-white font-medium' : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    Annotated Skeleton
                  </button>
                  <button
                    onClick={() => setShowAnnotated(false)}
                    className={`px-2.5 py-1 rounded-md transition-colors ${
                      !showAnnotated ? 'bg-blue-600 text-white font-medium' : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    Original Photo
                  </button>
                </div>
              )}
            </div>

            {/* Image Preview & Skeleton Display */}
            <div className="relative rounded-xl overflow-hidden bg-black border border-slate-800 flex items-center justify-center min-h-[320px] max-h-[460px]">
              {isAnalyzing && (
                <div className="absolute inset-0 bg-black/75 z-20 flex flex-col items-center justify-center gap-3 backdrop-blur-xs">
                  <RefreshCw className="w-8 h-8 text-blue-400 animate-spin" />
                  <span className="text-xs font-semibold text-slate-200">
                    Extracting skeletal landmarks & classifying pose...
                  </span>
                </div>
              )}

              {prediction && showAnnotated && prediction.annotated_image_base64 ? (
                <img
                  src={prediction.annotated_image_base64}
                  alt="Annotated Pose Skeleton"
                  className="w-full h-full object-contain max-h-[440px]"
                />
              ) : imagePreviewUrl ? (
                <img
                  src={imagePreviewUrl}
                  alt="Raw Posture Frame"
                  className="w-full h-full object-contain max-h-[440px]"
                />
              ) : (
                <div className="p-8 text-center text-slate-500 space-y-2">
                  <Camera className="w-12 h-12 mx-auto text-slate-600 stroke-[1.5]" />
                  <p className="text-xs font-medium text-slate-400">
                    No photo captured or selected yet.
                  </p>
                  <p className="text-[11px] text-slate-500">
                    Use the controls on the left to take a photo with your phone or select an image to inspect.
                  </p>
                </div>
              )}
            </div>

            {/* Results Presentation */}
            {prediction && (
              <div className="space-y-4 pt-2">
                {/* Hero Activity Alert Card */}
                <div
                  className={`p-4 rounded-xl border flex items-center justify-between gap-4 ${
                    activeMeta?.bg || 'bg-blue-500/10'
                  } ${activeMeta?.border || 'border-blue-500/30'}`}
                >
                  <div className="flex items-center gap-3">
                    <div
                      className={`p-3 rounded-xl bg-black/40 border border-white/10 ${
                        activeMeta?.color || 'text-blue-400'
                      }`}
                    >
                      <ActiveIcon className="w-6 h-6" />
                    </div>
                    <div>
                      <span className="text-[11px] uppercase tracking-wider text-slate-400 font-semibold block">
                        Classified Activity
                      </span>
                      <h2 className="text-xl font-bold text-white flex items-center gap-2">
                        <span>{activeMeta?.label || prediction.predicted_label}</span>
                        {prediction.predicted_activity === 'fall' && (
                          <span className="px-2 py-0.5 rounded bg-rose-500 text-white text-xs font-bold animate-pulse">
                            EMERGENCY
                          </span>
                        )}
                      </h2>
                    </div>
                  </div>

                  <div className="text-right">
                    <span className="text-[11px] uppercase tracking-wider text-slate-400 font-semibold block">
                      Confidence
                    </span>
                    <span className="text-2xl font-black font-mono text-white">
                      {(prediction.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>

                {/* Probability Distribution */}
                <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-3">
                  <span className="text-xs font-semibold text-slate-300 block">
                    Softmax Class Probabilities
                  </span>
                  <div className="space-y-2">
                    {Object.entries(prediction.probabilities).map(([act, prob]) => {
                      const meta = ACTIVITY_METADATA[act] || {
                        label: act,
                        color: 'text-slate-400',
                      };
                      const pct = Math.round(prob * 100);
                      const isTop = act === prediction.predicted_activity;

                      return (
                        <div key={act} className="space-y-1">
                          <div className="flex justify-between text-xs">
                            <span className={isTop ? 'font-bold text-white' : 'text-slate-400'}>
                              {meta.label}
                            </span>
                            <span className="font-mono text-slate-300">{pct}%</span>
                          </div>
                          <div className="h-1.5 w-full bg-slate-950 rounded-full overflow-hidden">
                            <div
                              className={`h-full rounded-full transition-all duration-500 ${
                                isTop
                                  ? act === 'fall'
                                    ? 'bg-rose-500'
                                    : 'bg-blue-500'
                                  : 'bg-slate-700'
                              }`}
                              style={{ width: `${pct}%` }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Biomechanical Telemetry Grid */}
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                  <div className="p-3 rounded-xl bg-slate-900/40 border border-slate-800">
                    <span className="text-[11px] text-slate-400 block">Torso Inclination</span>
                    <div className="text-base font-bold text-white font-mono mt-0.5">
                      {prediction.biomechanics.torso_angle_deg}°
                    </div>
                    <span className="text-[10px] text-blue-400">
                      {prediction.biomechanics.orientation}
                    </span>
                  </div>

                  <div className="p-3 rounded-xl bg-slate-900/40 border border-slate-800">
                    <span className="text-[11px] text-slate-400 block">Knee Flexion Angle</span>
                    <div className="text-base font-bold text-white font-mono mt-0.5">
                      {prediction.biomechanics.knee_angle_deg}°
                    </div>
                    <span className="text-[10px] text-slate-400">
                      {prediction.biomechanics.knee_angle_deg > 140 ? 'Straight' : 'Bent / Seated'}
                    </span>
                  </div>

                  <div className="p-3 rounded-xl bg-slate-900/40 border border-slate-800">
                    <span className="text-[11px] text-slate-400 block">Silhouette Aspect (W/H)</span>
                    <div className="text-base font-bold text-white font-mono mt-0.5">
                      {prediction.biomechanics.aspect_ratio}
                    </div>
                    <span className="text-[10px] text-slate-400">
                      {prediction.biomechanics.aspect_ratio > 1.0 ? 'Horizontal' : 'Vertical'}
                    </span>
                  </div>

                  <div className="p-3 rounded-xl bg-slate-900/40 border border-slate-800">
                    <span className="text-[11px] text-slate-400 block">Stride Separation</span>
                    <div className="text-base font-bold text-white font-mono mt-0.5">
                      {prediction.biomechanics.stride_separation}
                    </div>
                    <span className="text-[10px] text-slate-400">Limb separation</span>
                  </div>

                  <div className="p-3 rounded-xl bg-slate-900/40 border border-slate-800">
                    <span className="text-[11px] text-slate-400 block">Ground Proximity</span>
                    <div className="text-base font-bold text-white font-mono mt-0.5">
                      {prediction.biomechanics.ground_proximity}
                    </div>
                    <span className="text-[10px] text-slate-400">Floor level proximity</span>
                  </div>

                  <div className="p-3 rounded-xl bg-slate-900/40 border border-slate-800">
                    <span className="text-[11px] text-slate-400 block">Detected Joints</span>
                    <div className="text-base font-bold text-emerald-400 font-mono mt-0.5">
                      {prediction.keypoints.length}
                    </div>
                    <span className="text-[10px] text-slate-400">Skeletal points</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Mobile Connect Modal */}
      <MobileConnectModal
        isOpen={isMobileModalOpen}
        onClose={() => setIsMobileModalOpen(false)}
        targetPath="/vision"
      />
    </div>
  );
};
