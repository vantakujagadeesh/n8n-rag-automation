// src/App.jsx
import { useEffect, useState } from 'react';
import { api } from './api/client';
import { useToast } from './hooks/useToast';
import { Sidebar } from './components/Sidebar';
import { ToastContainer } from './components/Toast';
import { Dashboard } from './pages/Dashboard';
import { Chat } from './pages/Chat';
import { Ingest } from './pages/Ingest';
import { Documents } from './pages/Documents';
import { Health } from './pages/Health';

const PAGE_TITLES = {
  dashboard: 'Dashboard',
  chat:      'Ask AI — Chat Interface',
  ingest:    'Ingest Documents',
  documents: 'Knowledge Base',
  health:    'System Health',
};

export default function App() {
  const [page, setPage]     = useState('dashboard');
  const [health, setHealth] = useState(null);
  const toast = useToast();

  // Poll health every 30s
  useEffect(() => {
    const fetch = () =>
      api.health().then(setHealth).catch(() => setHealth(null));
    fetch();
    const id = setInterval(fetch, 30_000);
    return () => clearInterval(id);
  }, []);

  const renderPage = () => {
    switch (page) {
      case 'dashboard': return <Dashboard health={health} setPage={setPage} />;
      case 'chat':      return <Chat toast={toast} />;
      case 'ingest':    return <Ingest toast={toast} />;
      case 'documents': return <Documents toast={toast} />;
      case 'health':    return <Health />;
      default:          return <Dashboard health={health} setPage={setPage} />;
    }
  };

  return (
    <div className="app-layout">
      <Sidebar page={page} setPage={setPage} health={health} />

      <div className="main-content">
        {/* Top bar */}
        <header className="topbar">
          <div className="topbar-title">{PAGE_TITLES[page]}</div>
          <div className="topbar-actions">
            <a
              href="http://localhost:8000/docs"
              target="_blank"
              rel="noopener noreferrer"
              className="btn btn-secondary btn-sm"
              id="swagger-link"
            >
              📖 API Docs
            </a>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                padding: '6px 12px',
                borderRadius: 6,
                background: 'var(--bg-card)',
                border: '1px solid var(--border)',
                fontSize: 12,
                color: 'var(--text-secondary)',
              }}
            >
              <span
                style={{
                  width: 8,
                  height: 8,
                  borderRadius: '50%',
                  background: health?.status === 'ok'
                    ? 'var(--success)'
                    : health
                    ? 'var(--danger)'
                    : 'var(--warning)',
                  animation: 'pulse 2s infinite',
                }}
              />
              {health?.status === 'ok'
                ? 'Connected'
                : health
                ? 'Degraded'
                : 'Connecting…'}
            </div>
          </div>
        </header>

        {/* Page content */}
        {renderPage()}
      </div>

      <ToastContainer toasts={toast.toasts} />
    </div>
  );
}
