import React, { useState } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AnalysisProvider } from './context/AnalysisContext';
import { Navbar } from './components/Navbar';
import { Sidebar } from './components/Sidebar';
import { DashboardPage } from './pages/DashboardPage';
import { AnalyzerPage } from './pages/AnalyzerPage';
import { ResultsPage } from './pages/ResultsPage';

export function App() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  const toggleSidebar = () => {
    setIsSidebarOpen((prev) => !prev);
  };

  return (
    <AnalysisProvider>
      <BrowserRouter>
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
                {/* 3 Main Pages for Mini-Project */}
                <Route path="/" element={<DashboardPage />} />
                <Route path="/analyzer" element={<AnalyzerPage />} />
                <Route path="/results" element={<ResultsPage />} />

                {/* Legacy route redirects to consolidated pages */}
                <Route path="/models" element={<Navigate to="/" replace />} />
                <Route path="/evaluation" element={<Navigate to="/results" replace />} />
                <Route path="/about" element={<Navigate to="/" replace />} />
                <Route path="*" element={<Navigate to="/" replace />} />
              </Routes>
            </main>
          </div>
        </div>
      </BrowserRouter>
    </AnalysisProvider>
  );
}

export default App;
