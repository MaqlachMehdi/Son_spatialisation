import { create } from "zustand";
import type { SoundAsset } from "../types";
import { fetchSounds } from "../utils/api";

// Source unique pour le catalogue de sons (partagé + imports personnels) —
// partagée entre SourceInspector (menu déroulant) et SettingsDock (import/
// suppression), pour qu'un import se reflète immédiatement dans le menu
// sans recharger la page.
interface SoundsState {
  sounds: SoundAsset[];
  loading: boolean;
  error: string | null;
  refreshSounds: () => Promise<void>;
}

export const useSoundsStore = create<SoundsState>((set) => ({
  sounds: [],
  loading: false,
  error: null,

  refreshSounds: async () => {
    set({ loading: true, error: null });
    try {
      const sounds = await fetchSounds();
      set({ sounds, loading: false });
    } catch (err) {
      set({ error: err instanceof Error ? err.message : String(err), loading: false });
    }
  },
}));
