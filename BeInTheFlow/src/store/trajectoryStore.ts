import { create } from "zustand";
import type { TrajectoryDTO, TrajectoryType, TrajectoryWaypoint } from "../types";
import { useSceneStore } from "./sceneStore";
import { sampleTrajectory } from "../utils/trajectorySampling";
import { threeToSofa } from "../utils/sofaCoords";

let nextId = 1;

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

  updateTrajectory: (id, patch) =>
    set((state) => ({
      trajectories: state.trajectories.map((t) => (t.id === id ? { ...t, ...patch } : t)),
    })),

  selectTrajectory: (id) => set({ selectedTrajectoryId: id }),

  // Assigne la trajectoire à la source actuellement sélectionnée dans le
  // panneau des sources (couplage volontaire entre les deux stores), et la
  // positionne immédiatement sur le premier point de la trajectoire.
  applyToSelectedSource: (id) => {
    const { selectedId, updateSource } = useSceneStore.getState();
    if (selectedId) {
      const trajectory = get().trajectories.find((t) => t.id === id);
      const firstPoint = trajectory ? sampleTrajectory(trajectory)[0] : undefined;
      if (firstPoint) {
        const { azimuth, elevation, distance } = threeToSofa(firstPoint);
        updateSource(selectedId, {
          trajectoryId: id,
          azimuth: Math.round(azimuth * 10) / 10,
          elevation: Math.round(elevation * 10) / 10,
          distance: Math.round(distance * 100) / 100,
        });
      } else {
        updateSource(selectedId, { trajectoryId: id });
      }
    }
    set({ selectedTrajectoryId: id });
  },

  playTrajectory: (id) => set({ playingTrajectoryId: id }),
  stopTrajectory: () => set({ playingTrajectoryId: null }),
}));

export type { TrajectoryType, TrajectoryWaypoint };
