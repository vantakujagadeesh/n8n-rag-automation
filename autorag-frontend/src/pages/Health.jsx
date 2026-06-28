// src/pages/Health.jsx
import { useEffect, useState } from 'react';
import { api } from '../api/client';

function ServiceCard({ name, svc }) {
  const ok = svc?.status === 'ok';
  return (
    <div className={`health-card ${ok ? 'ok' : 'error'}`}>
      <div className="health-name">{name}</div>
      <div className="health-status">{ok ? '✅' : '❌'}</div>
      <div style={{ fontWeight: 600, fontSize: 13, margin: '4px 0', color: ok ? 'var(--success)' : 'var(--danger)' }}>
        {svc?.status?.toUpperCase() || 'UNKNOWN'}
      </div>
      <div className="health-latency">{svc?.latency_ms ?? '—'} ms</div>
      {svc?.detail && (
        <div style={{ fontSize: 11, color: 'var(--danger)', marginTop: 8, wordBreak: 'break-all' }}>
          {svc.detail}
        </div>
      )}
    </div>
  );
}

export function Health() {
  const [health, setHealth]   = useState(null);
  const [loading, setLoading] = useState(true);
  const [lastCheck, setLastCheck] = useState(null);

  const check = async () => {
    setLoading(true);
    try {
      const h = await api.health();
      setHealth(h);
      setLastCheck(new Date());
    } catch (err) {
      setHealth({ status: 'error', services: {}, version: '—' });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    check();
    const interval = setInterval(check, 30_000); // auto-refresh every 30s
    return () => clearInterval(interval);
  }, []);

  const allOk = health?.status === 'ok';

  return (
    <div className="page">
      <div className="page-header">
        <div className="flex items-center justify-between">
          <div>
            <div className="page-title">System Health</div>
            <div className="page-subtitle">
              Live status of all AutoRAG services. Auto-refreshes every 30 seconds.
            </div>
          </div>
          <button className="btn btn-secondary" onClick={check} disabled={loading} id="health-refresh-btn">
            {loading ? <span className="spinner" /> : '🔄'} Refresh
          </button>
        </div>
      </div>

      {/* Overall banner */}
      {health && (
        <div style={{
          padding: '20px 24px',
          borderRadius: 'var(--radius-lg)',
          background: allOk ? 'var(--success-glow)' : 'var(--danger-glow)',
          border: `1px solid ${allOk ? 'rgba(16,185,129,0.4)' : 'rgba(239,68,68,0.4)'}`,
          display: 'flex',
          alignItems: 'center',
          gap: 16,
          marginBottom: 24,
        }}>
          <span style={{ fontSize: 36 }}>{allOk ? '🟢' : '🔴'}</span>
          <div>
            <div style={{ fontWeight: 700, fontSize: 18, color: allOk ? 'var(--success)' : 'var(--danger)' }}>
              {allOk ? 'All Systems Operational' : 'System Degraded'}
            </div>
            <div style={{ fontSize: 13, color: 'var(--text-secondary)', marginTop: 2 }}>
              API Version: <code>{health.version}</code>&nbsp;·&nbsp;
              Last checked: {lastCheck?.toLocaleTimeString() || '—'}
            </div>
          </div>
        </div>
      )}

      {/* Service cards */}
      {loading && !health ? (
        <div className="empty-state">
          <div className="spinner" style={{ width: 36, height: 36, borderWidth: 3 }} />
          <p style={{ marginTop: 16, color: 'var(--text-muted)' }}>Checking services…</p>
        </div>
      ) : (
        <div className="health-grid">
          {health?.services &&
            Object.entries(health.services).map(([name, svc]) => (
              <ServiceCard key={name} name={name} svc={svc} />
            ))}
        </div>
      )}

      {/* Tech stack info */}
      <div className="card mt-6">
        <div className="card-title">Technology Stack</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
          {[
            { icon: '⚡', name: 'FastAPI + Uvicorn',       role: 'API Layer' },
            { icon: '🔵', name: 'Qdrant Cloud',            role: 'Vector Database' },
            { icon: '🔴', name: 'Redis',                   role: 'Cache & Dedup' },
            { icon: '🐘', name: 'PostgreSQL',              role: 'Metadata & Logs' },
            { icon: '🧬', name: 'text-embedding-3-large',  role: 'Embeddings (3072d)' },
            { icon: '🤖', name: 'GPT-4o + Claude Sonnet',  role: 'LLM Generation' },
            { icon: '🔍', name: 'Qdrant Hybrid Search',    role: 'Dense + BM25 + RRF' },
            { icon: '⚖️', name: 'ms-marco-MiniLM',         role: 'Cross-Encoder Reranker' },
            { icon: '🔁', name: 'n8n Workflows',           role: 'Automation' },
          ].map((t) => (
            <div key={t.name} style={{
              padding: '14px',
              background: 'var(--bg-secondary)',
              borderRadius: 'var(--radius-md)',
              border: '1px solid var(--border)',
            }}>
              <div style={{ fontSize: 20, marginBottom: 8 }}>{t.icon}</div>
              <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 4 }}>{t.name}</div>
              <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{t.role}</div>
            </div>
          ))}
        </div>
      </div>

      {/* API endpoints */}
      <div className="card mt-6">
        <div className="card-title">API Endpoints</div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {[
            { method: 'POST', path: '/ingest',            desc: 'Ingest PDF, DOCX, or URL' },
            { method: 'POST', path: '/query',             desc: 'Query the knowledge base' },
            { method: 'GET',  path: '/documents',         desc: 'List all documents' },
            { method: 'DELETE', path: '/documents/{id}',  desc: 'Delete a document' },
            { method: 'GET',  path: '/health',            desc: 'Service health check' },
            { method: 'GET',  path: '/metrics',           desc: 'Prometheus metrics scrape' },
            { method: 'GET',  path: '/docs',              desc: 'Interactive Swagger UI' },
          ].map((e) => (
            <div key={e.path} style={{
              display: 'flex',
              alignItems: 'center',
              gap: 12,
              padding: '10px 0',
              borderBottom: '1px solid var(--border)',
              fontSize: 13,
            }}>
              <span className={`badge ${
                e.method === 'GET'    ? 'badge-success' :
                e.method === 'POST'   ? 'badge-accent'  :
                e.method === 'DELETE' ? 'badge-pdf'     : 'badge-docx'
              }`} style={{ minWidth: 60, justifyContent: 'center' }}>
                {e.method}
              </span>
              <code style={{ minWidth: 200 }}>{e.path}</code>
              <span style={{ color: 'var(--text-muted)' }}>{e.desc}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
