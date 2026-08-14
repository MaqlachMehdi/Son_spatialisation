import { useSceneStore } from "../store/sceneStore";
import { useTrajectoryStore } from "../store/trajectoryStore";
import { fetchWorkspace, saveWorkspace } from "./api";

// Ne sauvegarde pas à chaque mutation (des dizaines par seconde pendant un
// drag) : on attend une accalmie avant d'envoyer l'état complet.
const AUTOSAVE_DELAY_MS = 1500;

let saveTimer: ReturnType<typeof setTimeout> | null = null;
let unsubscribeScene: (() => void) | null = null;
let unsubscribeTrajectory: (() => void) | null = null;

function scheduleSave() {
  if (saveTimer) clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    const { sources } = useSceneStore.getState();
    const { trajectories } = useTrajectoryStore.getState();
    saveWorkspace(sources, trajectories).catch((err) => {
      console.error("Échec de la sauvegarde automatique :", err);
    });
  }, AUTOSAVE_DELAY_MS);
}

// À activer une fois l'utilisateur connecté (après le chargement initial de
// son espace de travail, pour ne pas se re-sauvegarder soi-même aussitôt).
export function enableAutosync(): void {
  if (unsubscribeScene) return; // déjà actif
  unsubscribeScene = useSceneStore.subscribe(scheduleSave);
  unsubscribeTrajectory = useTrajectoryStore.subscribe(scheduleSave);
}

export function disableAutosync(): void {
  unsubscribeScene?.();
  unsubscribeTrajectory?.();
  unsubscribeScene = null;
  unsubscribeTrajectory = null;
  if (saveTimer) {
    clearTimeout(saveTimer);
    saveTimer = null;
  }
}

export async function loadWorkspace(): Promise<void> {
  const { sources, trajectories } = await fetchWorkspace();
  useSceneStore.getState().hydrateSources(sources);
  useTrajectoryStore.getState().hydrateTrajectories(trajectories);
}

export function resetWorkspace(): void {
  useSceneStore.getState().resetToDefault();
  useTrajectoryStore.getState().resetToDefault();
}
