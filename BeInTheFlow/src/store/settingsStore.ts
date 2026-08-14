import { create } from "zustand";

// HRTF sélectionnée dans le panneau de réglages. La mise à jour côté serveur
// (PUT /hrtfs/active) est déclenchée par SettingsDock, pas ici — ce store ne
// fait que refléter le choix courant pour l'affichage.
interface SettingsState {
  selectedHrtfId: string | null;
  setSelectedHrtf: (id: string) => void;
}

export const useSettingsStore = create<SettingsState>((set) => ({
  selectedHrtfId: null,
  setSelectedHrtf: (id) => set({ selectedHrtfId: id }),
}));
