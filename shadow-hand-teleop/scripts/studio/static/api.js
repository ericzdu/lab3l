// Fetch helpers. POSTs carry the per-start token the server put in the page.
const token = document.querySelector('meta[name="studio-token"]').content;

export async function getJSON(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) throw new Error(`${path}: ${response.status}`);
  return response.json();
}

export async function post(path, body = {}) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Studio-Token": token },
    body: JSON.stringify(body),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `${response.status} ${response.statusText}`);
  return data;
}

// Per-viewer conveniences only (the address never leaves this browser except to start the sim).
export function remember(key, value) {
  try { localStorage.setItem(`lab3l.${key}`, JSON.stringify(value)); } catch { /* private window */ }
}

export function recall(key, fallback) {
  try {
    const value = localStorage.getItem(`lab3l.${key}`);
    return value === null ? fallback : JSON.parse(value);
  } catch { return fallback; }
}
