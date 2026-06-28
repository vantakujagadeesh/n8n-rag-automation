// src/pages/Documents.jsx
import { useEffect, useState } from 'react';
import { api } from '../api/client';

export function Documents({ toast }) {
  const [docs, setDocs]       = useState(null);
  const [loading, setLoading] = useState(true);
  const [search, setSearch]   = useState('');
  const [deleting, setDeleting] = useState(null);

  const load = async () => {
    setLoading(true);
    try {
      const data = await api.listDocuments(200, 0);
      setDocs(data);
    } catch (err) {
      toast.error('Failed to load documents: ' + err.message);
      setDocs({ documents: [], total: 0 });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, []);

  const handleDelete = async (docId, filename) => {
    if (!window.confirm(`Delete "${filename}" from the knowledge base? This cannot be undone.`)) return;
    setDeleting(docId);
    try {
      const res = await api.deleteDocument(docId);
      toast.success(`"${filename}" deleted (${res.points_removed} vectors removed).`);
      setDocs((d) => ({
        ...d,
        documents: d.documents.filter((doc) => doc.id !== docId),
        total: d.total - 1,
      }));
    } catch (err) {
      toast.error('Delete failed: ' + err.message);
    } finally {
      setDeleting(null);
    }
  };

  const filtered = docs?.documents?.filter((d) =>
    d.filename?.toLowerCase().includes(search.toLowerCase())
  ) || [];

  return (
    <div className="page">
      <div className="page-header">
        <div className="flex items-center justify-between">
          <div>
            <div className="page-title">Knowledge Base</div>
            <div className="page-subtitle">
              {docs?.total ?? '…'} document{docs?.total !== 1 ? 's' : ''} indexed
            </div>
          </div>
          <button className="btn btn-secondary" onClick={load} id="docs-refresh-btn">
            🔄 Refresh
          </button>
        </div>
      </div>

      {/* Search */}
      <div className="mb-4">
        <input
          className="input"
          placeholder="🔍  Search documents…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          id="docs-search"
        />
      </div>

      {loading ? (
        <div className="empty-state">
          <div className="spinner" style={{ width: 36, height: 36, borderWidth: 3 }} />
          <p style={{ marginTop: 16, color: 'var(--text-muted)' }}>Loading documents…</p>
        </div>
      ) : filtered.length === 0 ? (
        <div className="empty-state">
          <div className="empty-icon">{search ? '🔍' : '📭'}</div>
          <div className="empty-title">
            {search ? 'No documents match your search' : 'No documents yet'}
          </div>
          <div className="empty-sub">
            {search
              ? 'Try a different search term.'
              : 'Upload your first document using the Ingest page.'}
          </div>
        </div>
      ) : (
        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Filename</th>
                <th>Type</th>
                <th>Chunks</th>
                <th>Doc ID</th>
                <th>Ingested</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((doc) => (
                <tr key={doc.id}>
                  <td>
                    <div style={{ fontWeight: 500, color: 'var(--text-primary)', fontSize: 13 }}>
                      {doc.filename}
                    </div>
                  </td>
                  <td>
                    <span className={`badge badge-${doc.file_type || 'url'}`}>
                      {doc.file_type || 'url'}
                    </span>
                  </td>
                  <td>
                    <span className="badge badge-accent font-mono">{doc.chunk_count}</span>
                  </td>
                  <td>
                    <code>{doc.id?.slice(0, 12)}…</code>
                  </td>
                  <td className="text-sm text-muted">
                    {doc.created_at ? new Date(doc.created_at).toLocaleString() : '—'}
                  </td>
                  <td>
                    <button
                      className="btn btn-danger btn-sm"
                      onClick={() => handleDelete(doc.id, doc.filename)}
                      disabled={deleting === doc.id}
                      id={`delete-doc-${doc.id}`}
                    >
                      {deleting === doc.id ? <span className="spinner" /> : '🗑 Delete'}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Summary cards */}
      {docs && docs.documents.length > 0 && (
        <div className="grid-2 mt-6">
          <div className="card">
            <div className="card-title">By Type</div>
            {Object.entries(
              docs.documents.reduce((acc, d) => {
                const t = d.file_type || 'url';
                acc[t] = (acc[t] || 0) + 1;
                return acc;
              }, {})
            ).map(([type, count]) => (
              <div key={type} style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 0', borderBottom: '1px solid var(--border)', fontSize: 13 }}>
                <span className={`badge badge-${type}`}>{type}</span>
                <span style={{ fontWeight: 700 }}>{count}</span>
              </div>
            ))}
          </div>
          <div className="card">
            <div className="card-title">Storage</div>
            <div style={{ fontSize: 28, fontWeight: 800, color: 'var(--accent-light)', marginBottom: 4 }}>
              {docs.documents.reduce((s, d) => s + (d.chunk_count || 0), 0).toLocaleString()}
            </div>
            <div style={{ fontSize: 13, color: 'var(--text-secondary)' }}>total vector chunks</div>
            <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 8 }}>
              ≈ {Math.round(docs.documents.reduce((s, d) => s + (d.chunk_count || 0), 0) * 3072 * 4 / 1024)} KB of vector data
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
