// Conversion entre la convention angulaire SOFA (azimut/élévation/distance,
// cf. src/scene/Trajectory.py : x=devant, y=gauche, z=haut) et les
// coordonnées cartésiennes Three.js (Y=haut, la caméra par défaut regarde
// vers -Z ; devant le listener = -Z, gauche = -X, droite = +X).

import * as THREE from "three";

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

// Cartésien SOFA (x=devant, y=gauche, z=haut, mètres) -> Three.js. Même
// transformation linéaire que sofaToThree ci-dessus (x₃=-y, y₃=z, z₃=-x),
// mais pour un point déjà cartésien plutôt que des coordonnées sphériques
// (az/el/distance) — utilisé pour positionner l'auditeur mobile (position
// suivie au sol par phoneMotion.ts, cf. ListenerHead.tsx / ListenerPath.tsx).
export function sofaCartesianToThree(x: number, y: number, z: number): Vec3 {
  return { x: -y, y: z, z: -x };
}

// Quaternion Three.js depuis (yaw, pitch, roll) en degrés, même convention
// que rotation_from_ypr côté backend (src/scene/geometry.py) : yaw>0 tourne
// vers la gauche, pitch>0 lève le nez, roll>0 penche l'oreille gauche vers
// le bas.
//
// Dérivation : conjuguer une rotation par un changement de repère orthogonal
// préserve l'angle et transforme l'axe par cette même transformation. Sous
// sofaCartesianToThree, l'axe de lacet (Z sofa, "haut") devient l'axe Y
// three.js, l'axe de tangage (Y sofa, gauche/droite) devient l'axe X
// three.js, et l'axe de roulis (X sofa, "devant") devient -Z three.js — d'où
// l'angle inversé sur roll pour rester sur l'axe +Z standard de three.js.
export function yprToThreeQuaternion(yawDeg: number, pitchDeg: number, rollDeg: number): THREE.Quaternion {
  const yaw = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), THREE.MathUtils.degToRad(yawDeg));
  const pitch = new THREE.Quaternion().setFromAxisAngle(
    new THREE.Vector3(1, 0, 0),
    THREE.MathUtils.degToRad(pitchDeg),
  );
  const roll = new THREE.Quaternion().setFromAxisAngle(
    new THREE.Vector3(0, 0, 1),
    THREE.MathUtils.degToRad(-rollDeg),
  );
  return yaw.multiply(pitch).multiply(roll);
}