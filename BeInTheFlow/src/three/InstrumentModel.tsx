import { useLayoutEffect, useMemo, useRef } from "react";
import { useLoader } from "@react-three/fiber";
import * as THREE from "three";
import { FBXLoader } from "three/examples/jsm/loaders/FBXLoader.js";
import { TDSLoader } from "three/examples/jsm/loaders/TDSLoader.js";
import type { InstrumentCatalogEntry } from "./instrumentCatalog";
import type { Vec3 } from "../utils/sofaCoords";

interface InstrumentModelProps {
  entry: InstrumentCatalogEntry;
  position: Vec3;
  selected: boolean;
}

// Même traitement visuel que ListenerHead (EvilScreamingManModel) : rendu
// filaire semi-transparent plutôt que les matériaux/textures d'origine, pour
// rester lisible et cohérent avec le reste de la scène.
const MODEL_COLOR = "#950787";

function useInstrumentObject(entry: InstrumentCatalogEntry): THREE.Object3D {
  // Les deux loaders retournent un THREE.Group ; le cache de useLoader est
  // partagé par URL donc on clone la hiérarchie pour que chaque source ait
  // sa propre instance dans la scène (un Object3D ne peut avoir qu'un parent).
  const source =
    entry.format === "fbx" ? useLoader(FBXLoader, entry.url) : useLoader(TDSLoader, entry.url);

  return useMemo(() => source.clone(true), [source]);
}

export default function InstrumentModel({ entry, position, selected }: InstrumentModelProps) {
  const object = useInstrumentObject(entry);
  const facingRef = useRef<THREE.Group>(null);

  const material = useMemo(
    () =>
      new THREE.MeshBasicMaterial({
        color: MODEL_COLOR,
        wireframe: true,
        transparent: true,
        opacity: selected ? 0.85 : 0.55,
        // Couleurs claires/saturées : avec le tone mapping ACES Filmic activé
        // par défaut sur le <Canvas> r3f, elles se font compresser vers le
        // blanc. On sort ce matériau du tone mapping pour qu'il garde
        // exactement la teinte demandée.
        toneMapped: false,
      }),
    [selected],
  );

  useLayoutEffect(() => {
    object.traverse((child) => {
      if (child instanceof THREE.Mesh) {
        child.material = material;
      }
    });
  }, [object, material]);

  // Le modèle doit toujours pointer vers le centre (l'auditeur, à l'origine),
  // même quand la source est déplacée : on réoriente le groupe à chaque
  // changement de position plutôt qu'une seule fois au montage.
  useLayoutEffect(() => {
    facingRef.current?.lookAt(0, 0, 0);
  }, [position.x, position.y, position.z]);

  return (
    <group ref={facingRef}>
      <primitive object={object} scale={entry.scale} rotation={entry.rotationOffset} raycast={() => null} />
    </group>
  );
}
