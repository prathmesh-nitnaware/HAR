import React, { useState } from 'react';
import { BrowserRouter, Routes, Route, Navigate, useLocation } from 'react-router-dom';
import { AnalysisProvider } from './context/AnalysisContext';
import { Navbar } from './components/Navbar';
import { Sidebar } from './components/Sidebar';
import { DashboardPage } from './pages/DashboardPage';
import { AnalyzerPage } from './pages/AnalyzerPage';
import { ResultsPage } from './pages/ResultsPage';
import { LiveStreamPage } from './pages/LiveStreamPage';
import { VisionPage } from './pages/VisionPage';
import { MobilePage } from './pages/MobilePage';

function AppLayout() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const location = useLocation();
  const isMobileRoute = location.pathname === '/mobile' || location.pathname === '/phone';

  const toggleSidebar = () => {
    setIsSidebarOpen((prev) => !prev);
  };

  if (isMobileRoute) {
    return (
      <Routes>
        <Route path="/mobile" element={<MobilePage />} />
        <Route path="/phone" element={<MobilePage />} />
      </Routes>
    );
  }

  return (
    <div className="min-h-screen bg-[#090d16] text-slate-100 flex flex-col font-sans selection:bg-blue-600 selection:text-white">
      <Navbar
        onToggleSidebar={toggleSidebar}
        isSidebarOpen={isSidebarOpen}
      />

      <div className="flex-1 flex overflow-hidden">
        <Sidebar
          isOpen={isSidebarOpen}
          onToggle={toggleSidebar}
        />

        <main className="flex-1 overflow-y-auto px-4 py-6 sm:px-6 lg:px-8">
          <Routes>
            {/* Main Navigation Pages */}
            <Route path="/" element={<DashboardPage />} />
            <Route path="/stream" element={<LiveStreamPage />} />
            <Route path="/vision" element={<VisionPage />} />
            <Route path="/analyzer" element={<AnalyzerPage />} />
            <Route path="/results" element={<ResultsPage />} />

            {/* Mobile Companion routes */}
            <Route path="/mobile" element={<MobilePage />} />
            <Route path="/phone" element={<MobilePage />} />

            {/* Legacy route redirects */}
            <Route path="/camera" element={<Navigate to="/vision" replace />} />
            <Route path="/photo" element={<Navigate to="/vision" replace />} />
            <Route path="/live" element={<Navigate to="/stream" replace />} />
            <Route path="/models" element={<Navigate to="/" replace />} />
            <Route path="/evaluation" element={<Navigate to="/results" replace />} />
            <Route path="/about" element={<Navigate to="/" replace />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>
    </div>
  );
}

export function App() {
  return (
    <AnalysisProvider>
      <BrowserRouter>
        <AppLayout />
      </BrowserRouter>
    </AnalysisProvider>
  );
}

export default App;
