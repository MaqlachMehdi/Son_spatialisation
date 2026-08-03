import { create } from "zustand";
import type { SoundSourceDTO } from "../types";

const PALETTE = ["#e57373", "#64b5f6", "#81c784", "#ffb74d", "#ba68c8", "#4db6ac"];

let nextId = 1;

function makeSource(overrides: Partial<SoundSourceDTO> = {}): SoundSourceDTO {
  const id = `source-${nextId}`;
  const color = PALETTE[(nextId - 1) % PALETTE.length];
  nextId += 1;
  return {
    id,
    label: `Source ${id.split("-")[1]}`,
    path: "",
    azimuth: 0,
    elevation: 0,
    distance: 2.06,
    gain: 1.0,
    color,
    ...overrides,
  };
}

interface SceneState {
  sources: SoundSourceDTO[];
  selectedId: string | null;
  addSource: () => void;
  removeSource: (id: string) => void;
  selectSource: (id: string | null) => void;
  updateSource: (id: string, patch: Partial<SoundSourceDTO>) => void;
}

export const useSceneStore = create<SceneState>((set) => ({
  sources: [makeSource({ azimuth: 30, elevation: 0 })],
  selectedId: null,

  addSource: () =>
    set((state) => {
      const source = makeSource();
      return { sources: [...state.sources, source], selectedId: source.id };
    }),

  removeSource: (id) =>
    set((state) => ({
      sources: state.sources.filter((s) => s.id !== id),
      selectedId: state.selectedId === id ? null : state.selectedId,
    })),

  selectSource: (id) => set({ selectedId: id }),

  updateSource: (id, patch) =>
    set((state) => ({
      sources: state.sources.map((s) => (s.id === id ? { ...s, ...patch } : s)),
    })),
}));