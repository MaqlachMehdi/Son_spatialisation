// Conversion entre la convention angulaire SOFA (azimut/élévation/distance,
// cf. src/scene/Trajectory.py : x=devant, y=gauche, z=haut) et les
// coordonnées cartésiennes Three.js (Y=haut, la caméra par défaut regarde
// vers -Z ; devant le listener = -Z, gauche = -X, droite = +X).

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

export function sofaToThree(azimuthDeg: number, elevationDeg: number, distance: number): Vec3 {
  const az = (azimuthDeg * Math.PI) / 180;
  const el = (elevationDeg * Math.PI) / 180;
  const cosEl = Math.cos(el);
  return {
    x: -distance * cosEl * Math.sin(az),
    y: distance * Math.sin(el),
    z: -distance * cosEl * Math.cos(az),
  };
}

export function threeToSofa(pos: Vec3): { azimuth: number; elevation: number; distance: number } {
  const distance = Math.sqrt(pos.x * pos.x + pos.y * pos.y + pos.z * pos.z);
  if (distance < 1e-6) {
    return { azimuth: 0, elevation: 0, distance: 0 };
  }
  const elevation = (Math.asin(pos.y / distance) * 180) / Math.PI;
  const azimuthRad = Math.atan2(-pos.x, -pos.z);
  const azimuth = (((azimuthRad * 180) / Math.PI) + 360) % 360;
  return { azimuth, elevation, distance };
}