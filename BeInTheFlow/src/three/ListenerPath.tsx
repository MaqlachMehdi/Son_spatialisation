import { Line } from "@react-three/drei";
import { useListenerStore } from "../store/listenerStore";
import { sofaCartesianToThree } from "../utils/sofaCoords";

// Trace le chemin parcouru par l'auditeur pendant une capture "auditeur
// dynamique" — un point par pas détecté (cf. phoneMotion.ts), pas par
// échantillon d'orientation : le nombre de pas reste raisonnable à afficher,
// contrairement à l'historique complet des échantillons d'orientation.
//
// Contrairement à ListenerHead (useFrame + lecture directe du store, pour
// l'animation temps réel), ce composant se ré-abonne normalement à `path` :
// il ne change qu'une fois par pas, un re-render React à cette fréquence
// n'a aucun impact perceptible.
export default function ListenerPath() {
  const path = useListenerStore((s) => s.path);
  if (path.length < 2) return null;

  const points = path.map((wp) => {
    const p = sofaCartesianToThree(wp.x, wp.y, 0);
    return [p.x, p.y, p.z] as [number, number, number];
  });

  return <Line points={points} color="#2196f3" lineWidth={2} raycast={() => null} />;
}
