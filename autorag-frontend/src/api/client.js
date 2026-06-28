// src/api/client.js
import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const client = axios.create({
  baseURL: API_BASE,
  timeout: 120_000,
  headers: { 'Content-Type': 'application/json' },
});

// Response interceptor — normalize errors
client.interceptors.response.use(
  (res) => res,
  (err) => {
    const detail =
      err.response?.data?.detail ||
      err.response?.data?.error ||
      err.message ||
      'Unknown error';
    return Promise.reject(new Error(detail));
  }
);

export const api = {
  // Health
  health: () => client.get('/health').then((r) => r.data),

  // Query
  query: (question, top_k = 5) =>
    client.post('/query', { question, top_k }).then((r) => r.data),

  // Ingest
  ingestFile: (file) => {
    const form = new FormData();
    form.append('file', file);
    return client
      .post('/ingest', form, { headers: { 'Content-Type': 'multipart/form-data' } })
      .then((r) => r.data);
  },

  ingestUrl: (url) => {
    const form = new FormData();
    form.append('file_url', url);
    return client
      .post('/ingest', form, { headers: { 'Content-Type': 'multipart/form-data' } })
      .then((r) => r.data);
  },

  // Documents
  listDocuments: (limit = 50, offset = 0) =>
    client.get('/documents', { params: { limit, offset } }).then((r) => r.data),

  deleteDocument: (docId) =>
    client.delete(`/documents/${docId}`).then((r) => r.data),

  // Metrics
  metrics: () => client.get('/metrics', { headers: { Accept: 'text/plain' } }).then((r) => r.data),
};

export default api;
