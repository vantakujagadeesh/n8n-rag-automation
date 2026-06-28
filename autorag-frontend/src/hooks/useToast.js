// src/hooks/useToast.js
import { useState, useCallback } from 'react';

let _id = 0;

export function useToast() {
  const [toasts, setToasts] = useState([]);

  const addToast = useCallback((message, type = 'info', duration = 4000) => {
    const id = ++_id;
    setToasts((t) => [...t, { id, message, type }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), duration);
  }, []);

  const success = (msg) => addToast(msg, 'success');
  const error   = (msg) => addToast(msg, 'error');
  const info    = (msg) => addToast(msg, 'info');

  return { toasts, success, error, info };
}
