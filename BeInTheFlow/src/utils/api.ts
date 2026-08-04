import type { RenderRequest, SoundAsset } from "../types";

const API_BASE = "http://localhost:8000";

export async function fetchSounds(): Promise<SoundAsset[]> {
  const res = await fetch(`${API_BASE}/sounds`);
  if (!res.ok) {
    throw new Error(`GET /sounds a échoué : ${res.status}`);
  }
  return res.json();
}

export async function renderScene(request: RenderRequest): Promise<Blob> {
  const res = await fetch(`${API_BASE}/render`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`POST /render a échoué (${res.status}) : ${detail}`);
  }
  return res.blob();
}