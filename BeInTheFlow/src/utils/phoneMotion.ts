import type { ListenerPose, ListenerWaypoint } from "../types";

// phoneMotion.ts
// --------------
// Capture du mouvement du téléphone pour piloter un auditeur mobile :
// orientation (boussole + inclinaison, via DeviceOrientationEvent) et
// détection de pas (accéléromètre, via DeviceMotionEvent), combinées pour
// approximer un déplacement dans la pièce.
//
// Pourquoi pas une double intégration de l'accélération pour la position ?
// Ça dérive en quelques secondes seulement (l'erreur croît de façon
// quadratique avec le bruit du capteur) — inutilisable au-delà d'un tout
// petit déplacement. Le comptage de pas (longueur moyenne fixe × direction
// courante) est moins précis mais ne dérive pas dans le temps.
//
// Convention de sortie (yaw, pitch, roll) : identique à rotation_from_ypr
// côté backend (src/scene/geometry.py) — yaw>0 tourne vers la gauche,
// pitch>0 lève le nez, roll>0 penche l'oreille gauche vers le bas.
// yaw est relatif au cap au moment du démarrage (le cap de départ devient
// l'azimut 0° de l'app), pas au nord magnétique — cohérent avec la
// convention SOFA du reste de l'app, où l'auditeur est censé démarrer face
// à ses sources plutôt que face au nord.

const STEP_LENGTH_M = 0.7; // longueur de pas moyenne — approximation fixe, pas mesurée individuellement
const STEP_THRESHOLD_MS2 = 1.5; // écart à la ligne de base pour détecter un pas — point de départ, à ajuster selon le téléphone
const STEP_REFRACTORY_MS = 300; // anti-rebond : un seul pas compté par intervalle (~3.3 pas/s max)
const BASELINE_EMA_ALPHA = 0.1; // vitesse d'adaptation de la ligne de base (gravité + inclinaison du téléphone)

export interface ListenerCaptureHandle {
  stop: () => void;
}

// Sous-ensemble minimal des APIs non-standard iOS Safari utilisées ici —
// absentes de lib.dom.d.ts, d'où ces types locaux plutôt que `any`.
interface PermissionRequestingEventConstructor {
  requestPermission?: () => Promise<"granted" | "denied">;
}

interface IOSOrientationEvent extends DeviceOrientationEvent {
  webkitCompassHeading?: number;
}

async function requestSensorPermission(ctor: unknown, sensorLabel: string): Promise<void> {
  const typed = ctor as PermissionRequestingEventConstructor;
  if (typeof typed.requestPermission !== "function") return; // Android, desktop... : rien à demander
  const result = await typed.requestPermission();
  if (result !== "granted") {
    throw new Error(`Permission d'accès ${sensorLabel} refusée.`);
  }
}

// Cap en degrés, sens horaire depuis le nord (convention boussole).
// webkitCompassHeading (iOS Safari) est déjà dans cette convention ; "alpha"
// (spec W3C, autres navigateurs) tourne dans l'autre sens, d'où le (360 - alpha).
function readCompassHeading(event: DeviceOrientationEvent): number | null {
  const heading = (event as IOSOrientationEvent).webkitCompassHeading;
  if (typeof heading === "number") return heading;
  if (event.alpha === null) return null;
  return (360 - event.alpha) % 360;
}

// Ramène un angle en degrés dans [-180, 180).
function wrapDegrees(deg: number): number {
  return ((deg + 180) % 360 + 360) % 360 - 180;
}

/**
 * Démarre la capture. Demande les permissions nécessaires (iOS 13+ — doit
 * être appelé depuis un gestionnaire de clic, sinon Safari refuse), puis
 * s'abonne aux capteurs jusqu'à l'appel de stop().
 *
 * @param onPose      Appelé à chaque échantillon d'orientation (fréquent,
 *                     pour une rotation de tête fluide en temps réel).
 * @param onWaypoint   Appelé à chaque pas détecté (position mise à jour) —
 *                     plus rare, un point par pas plutôt que par échantillon
 *                     d'orientation.
 */
export async function startListenerCapture(
  onPose: (pose: ListenerPose) => void,
  onWaypoint: (waypoint: ListenerWaypoint) => void,
): Promise<ListenerCaptureHandle> {
  await requestSensorPermission(DeviceOrientationEvent, "à l'orientation");
  await requestSensorPermission(DeviceMotionEvent, "au mouvement");

  const startTime = performance.now();
  let headingOffset: number | null = null;
  let yaw = 0;
  let pitch = 0;
  let roll = 0;
  let x = 0;
  let y = 0;
  let baseline: number | null = null;
  let lastStepAt = -Infinity;

  const handleOrientation = (event: DeviceOrientationEvent) => {
    const heading = readCompassHeading(event);
    if (heading === null) return;
    if (headingOffset === null) headingOffset = heading;

    yaw = wrapDegrees(heading - headingOffset);
    pitch = event.beta ?? 0;
    roll = -(event.gamma ?? 0);
    onPose({ x, y, yaw, pitch, roll });
  };

  const handleMotion = (event: DeviceMotionEvent) => {
    const a = event.accelerationIncludingGravity;
    if (!a || a.x == null || a.y == null || a.z == null) return;
    const magnitude = Math.sqrt(a.x * a.x + a.y * a.y + a.z * a.z);

    if (baseline === null) {
      baseline = magnitude;
      return;
    }
    const deviation = magnitude - baseline;
    baseline += BASELINE_EMA_ALPHA * (magnitude - baseline);

    const now = performance.now();
    if (deviation <= STEP_THRESHOLD_MS2 || now - lastStepAt <= STEP_REFRACTORY_MS) return;
    lastStepAt = now;

    const headingRad = (Math.PI / 180) * yaw;
    x += STEP_LENGTH_M * Math.cos(headingRad);
    y += STEP_LENGTH_M * Math.sin(headingRad);

    const waypoint: ListenerWaypoint = { t: (now - startTime) / 1000, x, y, yaw, pitch, roll };
    onWaypoint(waypoint);
    onPose(waypoint);
  };

  window.addEventListener("deviceorientation", handleOrientation);
  window.addEventListener("devicemotion", handleMotion);

  return {
    stop: () => {
      window.removeEventListener("deviceorientation", handleOrientation);
      window.removeEventListener("devicemotion", handleMotion);
    },
  };
}
