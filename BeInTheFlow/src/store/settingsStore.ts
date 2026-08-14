import { create } from "zustand";

// HRTF sélectionnée dans le panneau de réglages. Pour l'instant, ne change
// que cet état côté frontend — le rendu (/render) utilise toujours le SOFA
// chargé au démarrage du backend ; brancher la sélection dessus est une
// suite possible.
interface SettingsState {
  selectedHrtfId: string | null;
  setSelectedHrtf: (id: string) => void;
}

export const useSettingsStore = create<SettingsState>((set) => ({
  selectedHrtfId: null,
  setSelectedHrtf: (id) => set({ selectedHrtfId: id }),
}));
