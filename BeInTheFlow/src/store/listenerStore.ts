import { create } from "zustand";
import type { ListenerPose, ListenerWaypoint } from "../types";

const ORIGIN_POSE: ListenerPose = { x: 0, y: 0, yaw: 0, pitch: 0, roll: 0 };

interface ListenerState {
  // true pendant une capture "auditeur dynamique" (entre le démarrage et l'arrêt).
  dynamicMode: boolean;
  // Un point par pas détecté (cf. phoneMotion.ts) — pour tracer le chemin
  // parcouru, pas pour l'animation en temps réel (trop dense sinon).
  path: ListenerWaypoint[];
  // Pose courante — mise à jour à chaque échantillon d'orientation (fréquent),
  // lue par ListenerHead à chaque frame pour l'animation en temps réel.
  currentPose: ListenerPose;

  startCapture: () => void;
  stopCapture: () => void;
  updatePose: (pose: ListenerPose) => void;
  recordWaypoint: (waypoint: ListenerWaypoint) => void;
}

export const useListenerStore = create<ListenerState>((set) => ({
  dynamicMode: false,
  path: [],
  currentPose: ORIGIN_POSE,

  startCapture: () => set({ dynamicMode: true, path: [], currentPose: ORIGIN_POSE }),
  stopCapture: () => set({ dynamicMode: false }),

  updatePose: (pose) => set({ currentPose: pose }),

  recordWaypoint: (waypoint) => set((state) => ({ path: [...state.path, waypoint] })),
}));
