const BASE = '/api';

export async function createSession() {
  const res = await fetch(`${BASE}/session`, { method: 'POST' });
  if (!res.ok) throw new Error('Failed to create session');
  return res.json();
}

export async function sendMessage(sessionId, message) {
  const res = await fetch(`${BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, message }),
  });
  if (!res.ok) throw new Error('Failed to send message');
  return res.json();
}

export function getDownloadUrl(sessionId) {
  return `${BASE}/report/${sessionId}/download`;
}
