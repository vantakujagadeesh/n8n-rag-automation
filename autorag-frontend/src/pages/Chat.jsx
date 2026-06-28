// src/pages/Chat.jsx
import { useRef, useState, useEffect } from 'react';
import { api } from '../api/client';

function TypingIndicator() {
  return (
    <div className="message assistant">
      <div className="message-avatar">🤖</div>
      <div className="message-bubble" style={{ padding: '0', background: 'var(--bg-secondary)', border: '1px solid var(--border)' }}>
        <div className="typing-dots">
          <div className="typing-dot" />
          <div className="typing-dot" />
          <div className="typing-dot" />
        </div>
      </div>
    </div>
  );
}

function Message({ msg }) {
  if (msg.role === 'user') {
    return (
      <div className="message user">
        <div className="message-avatar">👤</div>
        <div>
          <div className="message-bubble">{msg.content}</div>
        </div>
      </div>
    );
  }

  return (
    <div className="message assistant">
      <div className="message-avatar">🤖</div>
      <div style={{ maxWidth: '75%' }}>
        <div className="message-bubble">
          <div style={{ lineHeight: 1.7 }}>{msg.content}</div>
          {msg.sources && msg.sources.length > 0 && (
            <div className="sources-list">
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4, fontWeight: 600 }}>
                📎 Sources
              </div>
              {msg.sources.map((src, i) => (
                <div key={i} className="source-chip">
                  <div className="source-num">{i + 1}</div>
                  <div>
                    <div className="source-file">{src.metadata?.filename || 'Unknown'}</div>
                    <div className="source-text">{src.metadata?.text_preview?.slice(0, 100)}…</div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
        {msg.meta && (
          <div className="message-meta">
            <span>🤖 {msg.meta.model_used}</span>
            <span>⏱ {msg.meta.latency_ms}ms</span>
            {msg.meta.hyde_used && <span>🔮 HyDE</span>}
            <span>🪙 {msg.meta.token_count} tokens</span>
          </div>
        )}
      </div>
    </div>
  );
}

const SUGGESTED = [
  'What documents are in the knowledge base?',
  'Summarize the main topics covered.',
  'What is the refund and return policy?',
  'Explain the key technical concepts.',
];

export function Chat({ toast }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput]       = useState('');
  const [loading, setLoading]   = useState(false);
  const [topK, setTopK]         = useState(5);
  const endRef = useRef(null);
  const textareaRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const sendMessage = async (text = input) => {
    const question = text.trim();
    if (!question || loading) return;
    setInput('');
    setMessages((m) => [...m, { role: 'user', content: question }]);
    setLoading(true);

    try {
      const res = await api.query(question, topK);
      setMessages((m) => [
        ...m,
        {
          role: 'assistant',
          content: res.answer,
          sources: res.sources,
          meta: {
            model_used: res.model_used,
            latency_ms: res.latency_ms,
            hyde_used: res.hyde_used,
            token_count: res.token_count,
          },
        },
      ]);
    } catch (err) {
      toast.error(err.message || 'Query failed');
      setMessages((m) => [
        ...m,
        { role: 'assistant', content: `❌ Error: ${err.message}`, sources: [] },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const clearChat = () => setMessages([]);

  return (
    <div className="page" style={{ paddingBottom: 0, height: 'calc(100vh - var(--topbar-height))' }}>
      <div className="chat-layout" style={{ height: '100%' }}>
        {/* Main chat area */}
        <div className="chat-main">
          <div className="chat-messages" id="chat-messages">
            {messages.length === 0 && (
              <div className="empty-state" style={{ flex: 1 }}>
                <div className="empty-icon">💬</div>
                <div className="empty-title">Ask anything from your knowledge base</div>
                <div className="empty-sub">
                  Upload documents first, then ask questions — you'll get cited answers backed by your own data.
                </div>
                <div className="flex gap-2 mt-4" style={{ flexWrap: 'wrap', justifyContent: 'center' }}>
                  {SUGGESTED.map((q) => (
                    <button
                      key={q}
                      className="btn btn-secondary btn-sm"
                      onClick={() => sendMessage(q)}
                      style={{ fontSize: 12 }}
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            )}
            {messages.map((msg, i) => (
              <Message key={i} msg={msg} />
            ))}
            {loading && <TypingIndicator />}
            <div ref={endRef} />
          </div>

          <div className="chat-input-area">
            <textarea
              ref={textareaRef}
              className="chat-textarea"
              placeholder="Ask a question… (Enter to send, Shift+Enter for newline)"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              rows={1}
              id="chat-input"
            />
            <button
              className="btn btn-primary"
              onClick={() => sendMessage()}
              disabled={loading || !input.trim()}
              id="chat-send-btn"
            >
              {loading ? <span className="spinner" /> : '→ Send'}
            </button>
          </div>
        </div>

        {/* Sidebar panel */}
        <div className="chat-sidebar">
          <div className="card">
            <div className="card-title">Settings</div>
            <label style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 6, display: 'block' }}>
              Top-K results: {topK}
            </label>
            <input
              type="range"
              min={1} max={20} step={1}
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              style={{ width: '100%', accentColor: 'var(--accent)' }}
              id="topk-slider"
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
              <span>1 (fast)</span>
              <span>20 (thorough)</span>
            </div>
            <button
              className="btn btn-secondary btn-sm mt-4"
              onClick={clearChat}
              style={{ width: '100%' }}
              id="chat-clear-btn"
            >
              🗑 Clear chat
            </button>
          </div>

          <div className="card">
            <div className="card-title">Pipeline</div>
            {[
              { label: 'HyDE', value: 'Enabled' },
              { label: 'Search', value: 'Hybrid' },
              { label: 'Reranker', value: 'MiniLM' },
              { label: 'Model', value: 'GPT-4o' },
            ].map((r) => (
              <div
                key={r.label}
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  fontSize: 12,
                  padding: '8px 0',
                  borderBottom: '1px solid var(--border)',
                }}
              >
                <span style={{ color: 'var(--text-muted)' }}>{r.label}</span>
                <span className="badge badge-accent">{r.value}</span>
              </div>
            ))}
          </div>

          <div className="card" style={{ flex: 1, overflowY: 'auto' }}>
            <div className="card-title">Session Stats</div>
            <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
              <div style={{ marginBottom: 8 }}>
                <span style={{ color: 'var(--text-muted)' }}>Questions asked</span>
                <div style={{ fontWeight: 700, fontSize: 20, color: 'var(--text-primary)' }}>
                  {messages.filter((m) => m.role === 'user').length}
                </div>
              </div>
              <div>
                <span style={{ color: 'var(--text-muted)' }}>Total sources cited</span>
                <div style={{ fontWeight: 700, fontSize: 20, color: 'var(--text-primary)' }}>
                  {messages.reduce((s, m) => s + (m.sources?.length || 0), 0)}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
