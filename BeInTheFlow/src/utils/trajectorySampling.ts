import * as THREE from "three";
import type { TrajectoryDTO } from "../types";
import { sofaToThree, type Vec3 } from "./sofaCoords";

const CIRCULAR_SEGMENTS = 72;
const ELLIPTICAL_SEGMENTS = 72;

// Position sur la trajectoire au temps normalisé t (0..1, une boucle
// complète). Fonction continue, réutilisée à la fois pour échantillonner le
// pointillé (sampleTrajectory) et pour animer une source (TrajectoryPlayer)
// — garantit que la source suit exactement la ligne affichée.
export function trajectoryPointAt(trajectory: TrajectoryDTO, t: number): Vec3 {
  const forward = ((t % 1) + 1) % 1;
  const u = trajectory.reverse ? 1 - forward : forward;

  switch (trajectory.type) {
    case "circular": {
      // Anneau à élévation fixe (axe par défaut = vertical), incliné autour
      // de son centre (axisAzimuth/axisElevation) et décalé par rapport à
      // l'auditeur (offsetAzimuth/Elevation/Distance).
      const axisDir = sofaToThree(trajectory.axisAzimuth, trajectory.axisElevation, 1);
      const tilt = new THREE.Quaternion().setFromUnitVectors(
        new THREE.Vector3(0, 1, 0),
        new THREE.Vector3(axisDir.x, axisDir.y, axisDir.z),
      );
      const offset = sofaToThree(trajectory.offsetAzimuth, trajectory.offsetElevation, trajectory.offsetDistance);

      const azimuth = 360 * u;
      const p = sofaToThree(azimuth, trajectory.centerElevation, trajectory.distance);
      const v = new THREE.Vector3(p.x, p.y, p.z).applyQuaternion(tilt);
      return { x: v.x + offset.x, y: v.y + offset.y, z: v.z + offset.z };
    }

    case "elliptical": {
      const angle = 2 * Math.PI * u;
      const azimuth = trajectory.centerAzimuth + trajectory.azAmplitude * Math.sin(angle);
      const elevation = trajectory.centerElevation + trajectory.elAmplitude * Math.cos(angle);
      return sofaToThree(azimuth, elevation, trajectory.distance);
    }

    case "points": {
      const n = trajectory.points.length;
      if (n === 0) return { x: 0, y: 0, z: 0 };
      if (n === 1) {
        const p = trajectory.points[0];
        return sofaToThree(p.azimuth, p.elevation, p.distance);
      }
      const segT = u * n;
      const i = Math.floor(segT) % n;
      const frac = segT - Math.floor(segT);
      const a = trajectory.points[i];
      const b = trajectory.points[(i + 1) % n];
      const pa = sofaToThree(a.azimuth, a.elevation, a.distance);
      const pb = sofaToThree(b.azimuth, b.elevation, b.distance);
      return {
        x: pa.x + (pb.x - pa.x) * frac,
        y: pa.y + (pb.y - pa.y) * frac,
        z: pa.z + (pb.z - pa.z) * frac,
      };
    }

    default:
      return { x: 0, y: 0, z: 0 };
  }
}

// Échantillonne une trajectoire en une liste de points Three.js (xyz), pour
// affichage en pointillé dans la scène. Sémantique alignée sur les classes
// Python existantes (src/scene/Trajectory.py) :
//   circular   ≈ CircularTrajectory  (balayage azimut complet, el/dist fixes)
//   elliptical ≈ EllipseTrajectory   (az/el oscillent en quadrature)
//   points     ≈ CustomTrajectory    (waypoints reliés dans l'ordre, boucle fermée)
export function sampleTrajectory(trajectory: TrajectoryDTO): Vec3[] {
  switch (trajectory.type) {
    case "circular": {
      const points: Vec3[] = [];
      for (let i = 0; i <= CIRCULAR_SEGMENTS; i++) {
        points.push(trajectoryPointAt(trajectory, i / CIRCULAR_SEGMENTS));
      }
      return points;
    }

    case "elliptical": {
      const points: Vec3[] = [];
      for (let i = 0; i <= ELLIPTICAL_SEGMENTS; i++) {
        points.push(trajectoryPointAt(trajectory, i / ELLIPTICAL_SEGMENTS));
      }
      return points;
    }

    case "points": {
      if (trajectory.points.length === 0) return [];
      const points = trajectory.points.map((p) => sofaToThree(p.azimuth, p.elevation, p.distance));
      if (trajectory.points.length > 1) {
        points.push(points[0]); // boucle fermée
      }
      return points;
    }

    default:
      return [];
  }
}
