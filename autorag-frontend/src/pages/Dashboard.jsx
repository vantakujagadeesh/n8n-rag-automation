// src/pages/Dashboard.jsx
import { useEffect, useState } from 'react';
import { api } from '../api/client';

function StatCard({ icon, value, label, change, accent }) {
  return (
    <div className="stat-card">
      <div className="stat-icon">{icon}</div>
      <div className="stat-value" style={accent ? { color: accent } : {}}>
        {value ?? <span className="spinner" style={{ width: 20, height: 20 }} />}
      </div>
      <div className="stat-label">{label}</div>
      {change && <div className="stat-change">{change}</div>}
    </div>
  );
}

export function Dashboard({ health, setPage }) {
  const [docs, setDocs] = useState(null);

  useEffect(() => {
    api.listDocuments(200, 0)
      .then((d) => setDocs(d))
      .catch(() => setDocs({ documents: [], total: 0 }));
  }, []);

  const totalChunks = docs?.documents?.reduce((s, d) => s + (d.chunk_count || 0), 0) ?? null;

  const qdrantOk   = health?.services?.qdrant?.status === 'ok';
  const redisOk    = health?.services?.redis?.status === 'ok';
  const postgresOk = health?.services?.postgres?.status === 'ok';

  const allOk = qdrantOk && redisOk && postgresOk;

  return (
    <div className="page">
      {/* Welcome banner */}
      <div className="welcome-banner">
        <span className="welcome-emoji">🧠</span>
        <div className="welcome-title">Welcome to AutoRAG</div>
        <p className="welcome-sub">
          Your production-grade retrieval-augmented generation pipeline. Upload documents,
          ask questions, and get cited AI answers instantly.
        </p>
        <div className="flex gap-2 mt-4">
          <button className="btn btn-primary" onClick={() => setPage('chat')} id="dashboard-ask-btn">
            💬 Ask a Question
          </button>
          <button className="btn btn-secondary" onClick={() => setPage('ingest')} id="dashboard-ingest-btn">
            📁 Ingest Document
          </button>
        </div>
      </div>

      {/* Stats */}
      <div className="stats-grid">
        <StatCard
          icon="📄"
          value={docs?.total ?? null}
          label="Total Documents"
          accent="#818cf8"
        />
        <StatCard
          icon="🔢"
          value={totalChunks !== null ? totalChunks.toLocaleString() : null}
          label="Vector Chunks"
          accent="#34d399"
        />
        <StatCard
          icon="🔍"
          value={health ? (allOk ? 'Healthy' : 'Degraded') : null}
          label="System Status"
          accent={allOk ? '#34d399' : '#f87171'}
        />
        <StatCard
          icon="⚙️"
          value={health ? '3 / 3' : null}
          label="Services Active"
          accent="#f59e0b"
        />
      </div>

      {/* Services health quick view */}
      <div className="card">
        <div className="card-title">Service Health</div>
        {health ? (
          <div className="health-grid">
            {Object.entries(health.services || {}).map(([name, svc]) => (
              <div key={name} className={`health-card ${svc.status}`}>
                <div className="health-name">{name}</div>
                <div className="health-status">
                  {svc.status === 'ok' ? '✅' : '❌'}
                </div>
                <div className="health-latency">{svc.latency_ms} ms</div>
              </div>
            ))}
          </div>
        ) : (
          <div className="empty-state" style={{ padding: '24px' }}>
            <div className="spinner" />
            <span style={{ marginTop: 12, color: 'var(--text-muted)' }}>
              Connecting to services…
            </span>
          </div>
        )}
      </div>

      {/* Recent docs */}
      {docs && docs.documents.length > 0 && (
        <div className="card mt-6">
          <div className="flex items-center justify-between mb-4">
            <div className="card-title" style={{ marginBottom: 0 }}>Recent Documents</div>
            <button className="btn btn-secondary btn-sm" onClick={() => setPage('documents')}>
              View all →
            </button>
          </div>
          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>Filename</th>
                  <th>Type</th>
                  <th>Chunks</th>
                  <th>Ingested</th>
                </tr>
              </thead>
              <tbody>
                {docs.documents.slice(0, 5).map((doc) => (
                  <tr key={doc.id}>
                    <td className="truncate" style={{ maxWidth: 240 }}>{doc.filename}</td>
                    <td>
                      <span className={`badge badge-${doc.file_type || 'url'}`}>
                        {doc.file_type || 'url'}
                      </span>
                    </td>
                    <td className="font-mono">{doc.chunk_count}</td>
                    <td className="text-xs text-muted">
                      {doc.created_at ? new Date(doc.created_at).toLocaleDateString() : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Pipeline explanation */}
      <div className="card mt-6">
        <div className="card-title">How It Works</div>
        <div className="grid-2">
          {[
            { icon: '📄', title: 'Parse', desc: 'PDF, DOCX, and URL content extracted with PyMuPDF & Playwright.' },
            { icon: '✂️', title: 'Chunk', desc: '512-token windows with 50-token overlap for optimal context coverage.' },
            { icon: '🧬', title: 'Embed', desc: 'OpenAI text-embedding-3-large (3072 dims) powers semantic search.' },
            { icon: '🔍', title: 'Retrieve', desc: 'Hybrid dense + BM25 search fused with Reciprocal Rank Fusion.' },
            { icon: '⚖️', title: 'Rerank', desc: 'Cross-encoder ms-marco-MiniLM reranks top-20 to top-5 results.' },
            { icon: '💡', title: 'Generate', desc: 'GPT-4o generates cited answers; Claude Sonnet as fallback.' },
          ].map((step) => (
            <div key={step.title} style={{ display: 'flex', gap: 12, padding: '12px 0', borderBottom: '1px solid var(--border)' }}>
              <span style={{ fontSize: 20 }}>{step.icon}</span>
              <div>
                <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 4 }}>{step.title}</div>
                <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>{step.desc}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
