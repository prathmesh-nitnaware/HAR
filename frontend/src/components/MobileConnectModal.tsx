import React, { useState, useEffect } from 'react';
import { QRCodeSVG } from 'qrcode.react';
import {
  Smartphone,
  X,
  Copy,
  Check,
  Wifi,
  ExternalLink,
  Camera,
  Activity,
  Info,
  RefreshCw,
} from 'lucide-react';
import { apiService, getBaseUrl } from '../api/client';

interface MobileConnectModalProps {
  isOpen: boolean;
  onClose: () => void;
  targetPath?: '/mobile' | '/vision' | '/stream';
}

export const MobileConnectModal: React.FC<MobileConnectModalProps> = ({
  isOpen,
  onClose,
  targetPath = '/mobile',
}) => {
  const [localIp, setLocalIp] = useState<string>('10.154.15.179');
  const [detectedPort, setDetectedPort] = useState<string>('5173');
  const [copied, setCopied] = useState<boolean>(false);
  const [activeTab, setActiveTab] = useState<'/mobile' | '/vision' | '/stream'>(targetPath);
  const [isLoading, setIsLoading] = useState<boolean>(false);

  useEffect(() => {
    setActiveTab(targetPath);
  }, [targetPath]);

  useEffect(() => {
    if (!isOpen) return;

    if (typeof window !== 'undefined') {
      setDetectedPort(window.location.port || '5173');
      if (
        window.location.hostname &&
        window.location.hostname !== 'localhost' &&
        window.location.hostname !== '127.0.0.1'
      ) {
        setLocalIp(window.location.hostname);
      }
    }

    setIsLoading(true);
    apiService
      .getNetworkInfo()
      .then((info) => {
        if (info.local_ip && info.local_ip !== '127.0.0.1') {
          setLocalIp(info.local_ip);
        }
      })
      .catch(() => {
        // Fallback to window hostname or default IP
      })
      .finally(() => {
        setIsLoading(false);
      });
  }, [isOpen]);

  if (!isOpen) return null;

  const displayIp = localIp || (typeof window !== 'undefined' && window.location.hostname !== 'localhost' ? window.location.hostname : '10.154.15.179');
  const portPart = detectedPort ? `:${detectedPort}` : '';
  const mobileUrl = `http://${displayIp}${portPart}${activeTab}`;

  const handleCopy = () => {
    navigator.clipboard.writeText(mobileUrl);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md animate-fade-in">
      <div className="relative w-full max-w-lg bg-[#0f172a] border border-slate-700/80 rounded-2xl shadow-2xl overflow-hidden text-slate-100">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-slate-900/60">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl bg-blue-600/20 text-blue-400 border border-blue-500/30">
              <Smartphone className="w-5 h-5" />
            </div>
            <div>
              <h3 className="font-semibold text-lg text-white">Connect Physical Smartphone</h3>
              <p className="text-xs text-slate-400">Everything will be analyzed and shown live on this PC screen</p>
            </div>
          </div>
          <button
            onClick={onClose}
            aria-label="Close modal"
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          {/* Target Mode Selector */}
          <div className="flex p-1 bg-slate-950/80 rounded-xl border border-slate-800 gap-1">
            <button
              onClick={() => setActiveTab('/mobile')}
              className={`flex-1 flex items-center justify-center gap-1.5 py-2 px-2.5 rounded-lg text-xs font-semibold transition-all ${
                activeTab === '/mobile'
                  ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/20'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Smartphone className="w-3.5 h-3.5" />
              <span>Mobile Companion</span>
            </button>
            <button
              onClick={() => setActiveTab('/vision')}
              className={`flex-1 flex items-center justify-center gap-1.5 py-2 px-2.5 rounded-lg text-xs font-medium transition-all ${
                activeTab === '/vision'
                  ? 'bg-purple-600 text-white shadow-lg shadow-purple-500/20'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Camera className="w-3.5 h-3.5" />
              <span>Camera Page</span>
            </button>
            <button
              onClick={() => setActiveTab('/stream')}
              className={`flex-1 flex items-center justify-center gap-1.5 py-2 px-2.5 rounded-lg text-xs font-medium transition-all ${
                activeTab === '/stream'
                  ? 'bg-emerald-600 text-white shadow-lg shadow-emerald-500/20'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Activity className="w-3.5 h-3.5" />
              <span>Sensors Page</span>
            </button>
          </div>

          {/* QR Code Card */}
          <div className="flex flex-col sm:flex-row items-center gap-6 p-4 rounded-xl bg-slate-900/50 border border-slate-800/80">
            <div className="p-3 bg-white rounded-xl shadow-md shrink-0">
              <QRCodeSVG
                value={mobileUrl}
                size={148}
                level="M"
                includeMargin={false}
              />
            </div>

            <div className="flex-1 space-y-3 text-center sm:text-left">
              <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-xs font-medium">
                <Wifi className="w-3.5 h-3.5 animate-pulse" />
                <span>Same Wi-Fi Network Required</span>
              </div>
              <p className="text-xs text-slate-300 leading-relaxed">
                Scan with your phone's native camera or QR scanner app to open directly on your mobile browser.
              </p>
              <div className="text-[11px] text-slate-500">
                Opening:{' '}
                <span className="font-mono text-blue-300">
                  {activeTab === '/vision' ? 'Camera & Photo Activity Detector' : 'Live Inertial Sensor Stream'}
                </span>
              </div>
            </div>
          </div>

          {/* Direct URL Box */}
          <div className="space-y-2">
            <label className="text-xs font-medium text-slate-300 flex items-center justify-between">
              <span>Mobile Browser URL</span>
              <span className="text-[11px] text-slate-500">Tap to copy</span>
            </label>
            <div className="flex items-center gap-2">
              <div className="flex-1 px-3 py-2.5 bg-slate-950 border border-slate-800 rounded-xl font-mono text-xs text-blue-300 truncate">
                {mobileUrl}
              </div>
              <button
                onClick={handleCopy}
                className="px-3 py-2.5 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-xl text-xs font-medium transition-colors flex items-center gap-1.5 shrink-0 border border-slate-700/60"
              >
                {copied ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
                <span>{copied ? 'Copied!' : 'Copy'}</span>
              </button>
            </div>
          </div>

          {/* Manual IP Adjustment if Needed */}
          <div className="pt-2 border-t border-slate-800/80">
            <div className="flex items-center justify-between text-xs text-slate-400">
              <span className="flex items-center gap-1.5">
                <Info className="w-3.5 h-3.5 text-slate-500" />
                Custom LAN IP:
              </span>
              <input
                type="text"
                value={localIp}
                onChange={(e) => setLocalIp(e.target.value.trim())}
                placeholder="e.g. 192.168.1.5"
                className="w-36 px-2.5 py-1 bg-slate-950 border border-slate-800 rounded-lg text-xs font-mono text-slate-200 focus:outline-none focus:border-blue-500"
              />
            </div>
          </div>

          {/* Quick Guide */}
          <div className="p-3 bg-blue-950/20 border border-blue-900/40 rounded-xl text-xs text-slate-300 space-y-1">
            <div className="font-semibold text-blue-300">📱 Mobile Companion Mode:</div>
            <ul className="list-disc list-inside space-y-0.5 text-slate-400 text-[11px]">
              <li>
                <strong>Wireless Camera Scanner:</strong> Snap a photo or choose an image on your phone — it will immediately project onto this PC screen with skeletal pose analysis.
              </li>
              <li>
                <strong>50 Hz IMU Streamer:</strong> Stream live phone gyroscope & accelerometer movement to this PC's live oscilloscope.
              </li>
            </ul>
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-3.5 bg-slate-900/40 border-t border-slate-800 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-white rounded-xl text-xs font-medium transition-colors"
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
};
