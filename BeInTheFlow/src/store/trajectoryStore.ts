import { create } from "zustand";
import type { TrajectoryDTO, TrajectoryType, TrajectoryWaypoint } from "../types";
import { useSceneStore } from "./sceneStore";
import { sampleTrajectory } from "../utils/trajectorySampling";
import { threeToSofa } from "../utils/sofaCoords";

let nextId = 1;

// Après un hydrateTrajectories() (chargement depuis le compte), les nouveaux
// id générés localement doivent continuer après le plus grand id restauré.
function bumpNextId(id: string) {
  const n = Number(id.split("-")[1]);
  if (Number.isFinite(n) && n >= nextId) nextId = n + 1;
}

// Position (az/el/distance, arrondie) du premier point de la trajectoire —
// utilisée pour accrocher une source à sa trajectoire, à l'association
// initiale comme à chaque modification ultérieure.
function firstPointPatch(trajectory: TrajectoryDTO) {
  const firstPoint = sampleTrajectory(trajectory)[0];
  if (!firstPoint) return null;
  const { azimuth, elevation, distance } = threeToSofa(firstPoint);
  return {
    azimuth: Math.round(azimuth * 10) / 10,
    elevation: Math.round(elevation * 10) / 10,
    distance: Math.round(distance * 100) / 100,
  };
}

// Repositionne toutes les sources associées à cette trajectoire sur son
// premier point — appelé à chaque édition (drag de poignée, champ modifié…)
// pour qu'une source liée reste toujours au contact de sa trajectoire.
function snapLinkedSources(trajectory: TrajectoryDTO) {
  const patch = firstPointPatch(trajectory);
  if (!patch) return;
  const { sources, updateSource } = useSceneStore.getState();
  sources.filter((s) => s.trajectoryId === trajectory.id).forEach((s) => updateSource(s.id, patch));
}

export function makeTrajectory(overrides: Partial<TrajectoryDTO> = {}): TrajectoryDTO {
  const id = `trajectory-${nextId}`;
  nextId += 1;
  return {
    id,
    name: `Trajectoire ${id.split("-")[1]}`,
    type: "circular",
    distance: 2.06,
    centerAzimuth: 0,
    centerElevation: 0,
    azAmplitude: 90,
    elAmplitude: 30,
    points: [],
    speed: 1,
    reverse: false,
    axisAzimuth: 0,
    axisElevation: 90,
    offsetAzimuth: 0,
    offsetElevation: 0,
    offsetDistance: 0,
    ...overrides,
  };
}

interface TrajectoryState {
  trajectories: TrajectoryDTO[];
  selectedTrajectoryId: string | null;
  playingTrajectoryId: string | null;
  addTrajectory: (overrides?: Partial<TrajectoryDTO>) => void;
  removeTrajectory: (id: string) => void;
  updateTrajectory: (id: string, patch: Partial<TrajectoryDTO>) => void;
  selectTrajectory: (id: string | null) => void;
  applyToSelectedSource: (id: string) => void;
  playTrajectory: (id: string) => void;
  stopTrajectory: () => void;
  // Remplace tout le state par ce qui vient d'être chargé depuis le compte.
  hydrateTrajectories: (trajectories: TrajectoryDTO[]) => void;
  // Repart de l'état par défaut (déconnexion) : aucune trajectoire.
  resetToDefault: () => void;
}

export const useTrajectoryStore = create<TrajectoryState>((set, get) => ({
  trajectories: [],
  selectedTrajectoryId: null,
  playingTrajectoryId: null,

  addTrajectory: (overrides) =>
    set((state) => {
      const trajectory = makeTrajectory(overrides);
      return {
        trajectories: [...state.trajectories, trajectory],
        selectedTrajectoryId: trajectory.id,
      };
    }),

  removeTrajectory: (id) =>
    set((state) => ({
      trajectories: state.trajectories.filter((t) => t.id !== id),
      selectedTrajectoryId: state.selectedTrajectoryId === id ? null : state.selectedTrajectoryId,
    })),

  updateTrajectory: (id, patch) => {
    set((state) => ({
      trajectories: state.trajectories.map((t) => (t.id === id ? { ...t, ...patch } : t)),
    }));
    const trajectory = get().trajectories.find((t) => t.id === id);
    if (trajectory) snapLinkedSources(trajectory);
  },

  selectTrajectory: (id) => set({ selectedTrajectoryId: id }),

  // Assigne la trajectoire à la source actuellement sélectionnée dans le
  // panneau des sources (couplage volontaire entre les deux stores), et la
  // positionne immédiatement sur le premier point de la trajectoire.
  applyToSelectedSource: (id) => {
    const { selectedId, updateSource } = useSceneStore.getState();
    if (selectedId) {
      const trajectory = get().trajectories.find((t) => t.id === id);
      const patch = trajectory ? firstPointPatch(trajectory) : null;
      updateSource(selectedId, patch ? { trajectoryId: id, ...patch } : { trajectoryId: id });
    }
    set({ selectedTrajectoryId: id });
  },

  playTrajectory: (id) => set({ playingTrajectoryId: id }),
  stopTrajectory: () => set({ playingTrajectoryId: null }),

  hydrateTrajectories: (trajectories) => {
    trajectories.forEach((t) => bumpNextId(t.id));
    set({ trajectories, selectedTrajectoryId: null, playingTrajectoryId: null });
  },

  resetToDefault: () => set({ trajectories: [], selectedTrajectoryId: null, playingTrajectoryId: null }),
}));

export type { TrajectoryType, TrajectoryWaypoint };
