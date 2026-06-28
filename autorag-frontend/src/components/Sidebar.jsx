// src/components/Sidebar.jsx
import { useState } from 'react';

const NAV_ITEMS = [
  { id: 'dashboard', label: 'Dashboard',  icon: '⚡' },
  { id: 'chat',      label: 'Ask AI',     icon: '💬' },
  { id: 'ingest',    label: 'Ingest Docs',icon: '📁' },
  { id: 'documents', label: 'Documents',  icon: '📚' },
  { id: 'health',    label: 'Health',     icon: '🩺' },
];

export function Sidebar({ page, setPage, health }) {
  const overall = health?.status;

  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <div className="logo-icon">🧠</div>
        <span className="logo-text">AutoRAG</span>
      </div>

      <nav className="sidebar-nav">
        {NAV_ITEMS.map((item) => (
          <button
            key={item.id}
            className={`nav-item ${page === item.id ? 'active' : ''}`}
            onClick={() => setPage(item.id)}
            id={`nav-${item.id}`}
          >
            <span className="nav-icon">{item.icon}</span>
            {item.label}
          </button>
        ))}
      </nav>

      <div className="sidebar-footer">
        <div className="status-badge">
          <span
            className={`status-dot ${
              !health ? 'loading' : overall === 'ok' ? '' : 'error'
            }`}
          />
          <span>
            {!health
              ? 'Connecting…'
              : overall === 'ok'
              ? 'All systems OK'
              : 'Services degraded'}
          </span>
        </div>
      </div>
    </aside>
  );
}
