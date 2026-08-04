// Champs alignés sur SoundSource (src/scene/Soundsource.py)
export interface SoundSourceDTO {
  id: string;
  label: string;
  path: string;
  azimuth: number;   // degrés, convention SOFA : 0°=devant, 90°=gauche
  elevation: number; // degrés, -90°=bas, 90°=dessus
  distance: number;  // mètres
  gain: number;      // gain linéaire
  color: string;
  trajectoryId?: string | null;
}

// Types de trajectoires — mêmes concepts que src/scene/Trajectory.py
// (CircularTrajectory / EllipseTrajectory / CustomTrajectory), pour l'instant
// utilisés uniquement pour la définition/visualisation côté frontend.
export type TrajectoryType = "circular" | "elliptical" | "points";

export interface TrajectoryWaypoint {
  azimuth: number;
  elevation: number;
  distance: number;
}

export interface TrajectoryDTO {
  id: string;
  name: string;
  type: TrajectoryType;
  distance: number;       // rayon (m), pour circular/elliptical
  centerAzimuth: number;  // degrés — centre, pour circular/elliptical
  centerElevation: number;
  azAmplitude: number;    // demi-amplitude azimut (elliptical uniquement)
  elAmplitude: number;    // demi-amplitude élévation (elliptical uniquement)
  points: TrajectoryWaypoint[]; // pour le type "points"

  // Vitesse de déplacement le long de la trajectoire (multiplicateur, 1 =
  // vitesse par défaut). Utilisé par la lecture animée (TrajectoryPlayer).
  speed: number;

  // "circular" uniquement — orientation de l'axe (pôle) du cercle, permet de
  // l'incliner librement au lieu de le forcer parallèle au sol.
  // Défaut (0°, 90°) = axe vertical = cercle horizontal, comme avant.
  axisAzimuth: number;
  axisElevation: number;

  // "circular" uniquement — décalage du centre du cercle par rapport à
  // l'auditeur (direction + distance). Défaut distance=0 = cercle centré
  // sur l'auditeur, comme avant.
  offsetAzimuth: number;
  offsetElevation: number;
  offsetDistance: number;
}

export interface SoundAsset {
  id: string;
  label: string;
  path: string;
}

export interface RenderRequest {
  sources: {
    path: string;
    azimuth: number;
    elevation: number;
    distance: number;
    gain: number;
    label: string;
  }[];
}