import type {
  AuthUser,
  HrtfAsset,
  RenderRequest,
  SoundAsset,
  SoundSourceDTO,
  TrajectoryDTO,
} from "../types";

const API_BASE = "http://localhost:8000";

// Messages fastapi-users (codes d'erreur stables, cf. sa doc) traduits pour
// l'UI plutôt que d'afficher le code brut à l'utilisateur.
const AUTH_ERROR_MESSAGES: Record<string, string> = {
  LOGIN_BAD_CREDENTIALS: "Email ou mot de passe incorrect.",
  LOGIN_USER_NOT_VERIFIED: "Ce compte n'est pas encore vérifié.",
  REGISTER_USER_ALREADY_EXISTS: "Un compte existe déjà avec cet email.",
  REGISTER_INVALID_PASSWORD: "Mot de passe invalide (trop court ou trop simple).",
};

async function parseAuthError(res: Response): Promise<string> {
  const body = await res.text();
  try {
    const json = JSON.parse(body);
    const code = typeof json.detail === "string" ? json.detail : json.detail?.code;
    if (code && AUTH_ERROR_MESSAGES[code]) return AUTH_ERROR_MESSAGES[code];
    if (typeof json.detail === "string") return json.detail;
  } catch {
    // pas du JSON — on retombe sur le message générique ci-dessous
  }
  return `Erreur (${res.status})`;
}

function mapUser(raw: {
  id: string;
  email: string;
  is_active: boolean;
  is_verified: boolean;
  display_name: string | null;
}): AuthUser {
  return {
    id: raw.id,
    email: raw.email,
    isActive: raw.is_active,
    isVerified: raw.is_verified,
    displayName: raw.display_name,
  };
}

// null = pas de session valide (visiteur anonyme), pas une erreur.
export async function fetchMe(): Promise<AuthUser | null> {
  const res = await fetch(`${API_BASE}/users/me`, { credentials: "include" });
  if (res.status === 401) return null;
  if (!res.ok) throw new Error(`GET /users/me a échoué : ${res.status}`);
  return mapUser(await res.json());
}

export async function registerUser(
  email: string,
  password: string,
  displayName: string,
): Promise<AuthUser> {
  const res = await fetch(`${API_BASE}/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({ email, password, display_name: displayName || null }),
  });
  if (!res.ok) throw new Error(await parseAuthError(res));
  return mapUser(await res.json());
}

// fastapi-users attend le flux OAuth2 "password" (form-urlencoded,
// champ "username"), pas du JSON — d'où le URLSearchParams ici.
export async function loginUser(email: string, password: string): Promise<void> {
  const body = new URLSearchParams({ username: email, password });
  const res = await fetch(`${API_BASE}/auth/cookie/login`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    credentials: "include",
    body,
  });
  if (!res.ok) throw new Error(await parseAuthError(res));
}

export async function logoutUser(): Promise<void> {
  await fetch(`${API_BASE}/auth/cookie/logout`, { method: "POST", credentials: "include" });
}

interface WorkspacePayload {
  sources: SoundSourceDTO[];
  trajectories: TrajectoryDTO[];
}

// Les champs de SoundSourceDTO/TrajectoryDTO correspondent déjà exactement
// aux alias camelCase exposés par /workspace (voir schemas/workspace.py côté
// backend) — aucun mapping de champ nécessaire dans un sens ni dans l'autre.
export async function fetchWorkspace(): Promise<WorkspacePayload> {
  const res = await fetch(`${API_BASE}/workspace`, { credentials: "include" });
  if (!res.ok) throw new Error(`GET /workspace a échoué : ${res.status}`);
  return res.json();
}

export async function saveWorkspace(sources: SoundSourceDTO[], trajectories: TrajectoryDTO[]): Promise<void> {
  const res = await fetch(`${API_BASE}/workspace`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({ sources, trajectories }),
  });
  if (!res.ok) throw new Error(`PUT /workspace a échoué : ${res.status}`);
}

export async function fetchSounds(): Promise<SoundAsset[]> {
  const res = await fetch(`${API_BASE}/sounds`);
  if (!res.ok) {
    throw new Error(`GET /sounds a échoué : ${res.status}`);
  }
  return res.json();
}

export async function fetchHrtfs(): Promise<HrtfAsset[]> {
  const res = await fetch(`${API_BASE}/hrtfs`);
  if (!res.ok) {
    throw new Error(`GET /hrtfs a échoué : ${res.status}`);
  }
  return res.json();
}

export async function renderScene(request: RenderRequest, signal?: AbortSignal): Promise<Blob> {
  const res = await fetch(`${API_BASE}/render`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
    signal,
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`POST /render a échoué (${res.status}) : ${detail}`);
  }
  return res.blob();
}