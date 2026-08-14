import { create } from "zustand";
import type { AuthUser } from "../types";
import { fetchMe, loginUser, logoutUser, registerUser } from "../utils/api";
import { disableAutosync, enableAutosync, loadWorkspace, resetWorkspace } from "../utils/workspaceSync";

type AuthStatus = "checking" | "anonymous" | "authenticated";

interface AuthState {
  user: AuthUser | null;
  status: AuthStatus;
  error: string | null;
  checkSession: () => Promise<void>;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, displayName: string) => Promise<void>;
  logout: () => Promise<void>;
  clearError: () => void;
}

// Charge l'espace de travail sauvegardé (sources + trajectoires) puis
// n'active la sauvegarde auto qu'après coup, pour ne pas se re-sauvegarder
// soi-même dès l'hydratation. Les erreurs de chargement n'empêchent pas la
// connexion elle-même de réussir — juste loggées.
async function syncWorkspaceAfterAuth() {
  try {
    await loadWorkspace();
  } catch (err) {
    console.error("Échec du chargement de l'espace de travail :", err);
  } finally {
    enableAutosync();
  }
}

// checkSession() est appelée au montage de l'app dans un useEffect — React
// StrictMode (dev) invoque les effets deux fois, donc deux checkSession()
// tournent en parallèle. Si la première (partie avant une éventuelle
// connexion manuelle rapide) répond APRÈS coup, elle écraserait le state
// "authenticated" tout juste posé par login()/register(). Un compteur de
// génération partagé par toutes les actions ci-dessous garantit qu'seul le
// dernier appel en date peut encore écrire le state.
let authCallId = 0;

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  status: "checking",
  error: null,

  // Appelée une fois au montage de l'app : restaure la session si le cookie
  // httpOnly posé par un précédent login est encore valide.
  checkSession: async () => {
    const callId = ++authCallId;
    try {
      const user = await fetchMe();
      if (callId !== authCallId) return; // supplantée par un appel plus récent
      set({ user, status: user ? "authenticated" : "anonymous" });
      if (user) await syncWorkspaceAfterAuth();
    } catch {
      if (callId !== authCallId) return;
      set({ user: null, status: "anonymous" });
    }
  },

  login: async (email, password) => {
    const callId = ++authCallId;
    set({ error: null });
    try {
      await loginUser(email, password);
      const user = await fetchMe();
      if (callId !== authCallId) return;
      set({ user, status: user ? "authenticated" : "anonymous" });
      if (user) await syncWorkspaceAfterAuth();
    } catch (err) {
      if (callId !== authCallId) return;
      set({ error: err instanceof Error ? err.message : String(err) });
      throw err;
    }
  },

  register: async (email, password, displayName) => {
    const callId = ++authCallId;
    set({ error: null });
    try {
      await registerUser(email, password, displayName);
      await loginUser(email, password);
      const user = await fetchMe();
      if (callId !== authCallId) return;
      set({ user, status: user ? "authenticated" : "anonymous" });
      if (user) await syncWorkspaceAfterAuth();
    } catch (err) {
      if (callId !== authCallId) return;
      set({ error: err instanceof Error ? err.message : String(err) });
      throw err;
    }
  },

  logout: async () => {
    authCallId++; // invalide toute vérification de session encore en vol
    disableAutosync();
    await logoutUser();
    resetWorkspace();
    set({ user: null, status: "anonymous" });
  },

  clearError: () => set({ error: null }),
}));
