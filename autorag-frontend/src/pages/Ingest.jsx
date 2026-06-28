// src/pages/Ingest.jsx
import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { api } from '../api/client';

const ACCEPTED_TYPES = {
  'application/pdf': ['.pdf'],
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
};

export function Ingest({ toast }) {
  const [uploading, setUploading]   = useState(false);
  const [progress, setProgress]     = useState(0);
  const [results, setResults]       = useState([]);
  const [urlInput, setUrlInput]     = useState('');
  const [urlLoading, setUrlLoading] = useState(false);

  const ingestFile = useCallback(async (file) => {
    setUploading(true);
    setProgress(10);

    // Fake progress ticks while waiting
    const tick = setInterval(() => setProgress((p) => Math.min(p + 8, 85)), 600);

    try {
      const res = await api.ingestFile(file);
      clearInterval(tick);
      setProgress(100);
      setResults((r) => [{ ...res, filename: file.name, ts: new Date().toISOString() }, ...r]);

      if (res.skipped) {
        toast.info(`"${file.name}" already ingested (duplicate).`);
      } else {
        toast.success(`"${file.name}" ingested — ${res.chunk_count} chunks stored.`);
      }
    } catch (err) {
      clearInterval(tick);
      toast.error(`Failed to ingest "${file.name}": ${err.message}`);
    } finally {
      setTimeout(() => {
        setUploading(false);
        setProgress(0);
      }, 800);
    }
  }, [toast]);

  const onDrop = useCallback((files) => {
    files.forEach((f) => ingestFile(f));
  }, [ingestFile]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPTED_TYPES,
    maxFiles: 5,
    maxSize: 50 * 1024 * 1024,
    disabled: uploading,
  });

  const handleUrlIngest = async () => {
    const url = urlInput.trim();
    if (!url) return;
    setUrlLoading(true);
    try {
      const res = await api.ingestUrl(url);
      setResults((r) => [{ ...res, filename: res.filename || url, ts: new Date().toISOString() }, ...r]);
      toast.success(`URL ingested — ${res.chunk_count} chunks stored.`);
      setUrlInput('');
    } catch (err) {
      toast.error(`URL ingest failed: ${err.message}`);
    } finally {
      setUrlLoading(false);
    }
  };

  return (
    <div className="page">
      <div className="page-header">
        <div className="page-title">Ingest Documents</div>
        <div className="page-subtitle">
          Upload PDF/DOCX files or provide a URL to add to the knowledge base.
        </div>
      </div>

      <div className="grid-2">
        {/* File upload */}
        <div>
          <div
            {...getRootProps()}
            className={`dropzone ${isDragActive ? 'active' : ''} ${uploading ? 'disabled' : ''}`}
            id="file-dropzone"
          >
            <input {...getInputProps()} id="file-input" />
            <div className="dropzone-icon">{uploading ? '⏳' : '📂'}</div>
            <div className="dropzone-title">
              {isDragActive ? 'Drop files here…' : 'Drag & drop or click to upload'}
            </div>
            <div className="dropzone-sub">PDF and DOCX up to 50 MB • Up to 5 files at once</div>
          </div>

          {uploading && (
            <div className="mt-4">
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 6, color: 'var(--text-muted)' }}>
                <span>Processing…</span>
                <span>{progress}%</span>
              </div>
              <div className="progress-bar">
                <div className="progress-fill" style={{ width: `${progress}%` }} />
              </div>
            </div>
          )}
        </div>

        {/* URL ingest */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div className="card-title">Ingest from URL</div>
          <div style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
            Provide a public URL to a PDF or web page to extract and index its content.
          </div>
          <input
            className="input"
            type="url"
            placeholder="https://example.com/document.pdf"
            value={urlInput}
            onChange={(e) => setUrlInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleUrlIngest()}
            id="url-input"
          />
          <button
            className="btn btn-primary"
            onClick={handleUrlIngest}
            disabled={urlLoading || !urlInput.trim()}
            id="url-ingest-btn"
          >
            {urlLoading ? <><span className="spinner" /> Processing…</> : '🌐 Ingest URL'}
          </button>
        </div>
      </div>

      {/* Results log */}
      {results.length > 0 && (
        <div className="card mt-6">
          <div className="card-title">Ingest Log</div>
          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>Filename</th>
                  <th>Status</th>
                  <th>Chunks</th>
                  <th>Doc ID</th>
                  <th>Latency</th>
                </tr>
              </thead>
              <tbody>
                {results.map((r, i) => (
                  <tr key={i}>
                    <td className="truncate" style={{ maxWidth: 200 }}>{r.filename}</td>
                    <td>
                      {r.skipped ? (
                        <span className="badge" style={{ background: 'rgba(245,158,11,0.15)', color: '#fbbf24' }}>
                          ⚠️ Duplicate
                        </span>
                      ) : (
                        <span className="badge badge-success">✅ Ingested</span>
                      )}
                    </td>
                    <td className="font-mono">{r.chunk_count}</td>
                    <td className="font-mono" style={{ fontSize: 11 }}>{r.doc_id?.slice(0, 12)}…</td>
                    <td className="font-mono">{r.latency_ms} ms</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Instructions */}
      <div className="card mt-6">
        <div className="card-title">What happens after upload?</div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {[
            ['🔒', 'Deduplication', 'SHA-256 hash check prevents re-indexing the same document.'],
            ['📄', 'Parsing',        'PyMuPDF extracts PDF text; python-docx handles Word files; Playwright scrapes URLs.'],
            ['✂️', 'Chunking',       '512-character overlapping windows split content for optimal retrieval.'],
            ['🧬', 'Embedding',      'OpenAI text-embedding-3-large creates 3072-dim dense vectors per chunk.'],
            ['💾', 'Storage',        'Vectors stored in Qdrant Cloud; metadata & audit trail in PostgreSQL.'],
          ].map(([icon, title, desc]) => (
            <div key={title} style={{ display: 'flex', gap: 12, alignItems: 'flex-start' }}>
              <span style={{ fontSize: 18, width: 28, flexShrink: 0 }}>{icon}</span>
              <div>
                <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 2 }}>{title}</div>
                <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>{desc}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
