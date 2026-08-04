import { useMemo } from "react";
import * as THREE from "three";
import { useGLTF } from "@react-three/drei";
import type { GLTF } from "three-stdlib";
import type { ThreeElements } from "@react-three/fiber";

type GLTFResult = GLTF & {
  nodes: {
    EvilScreaminManBaseMesh: THREE.Mesh;
  };
  materials: {
    EvilScrMan: THREE.MeshStandardMaterial;
  };
};

const HEAD_COLOR = "#226F60";

export function Model(props: ThreeElements["group"]) {
  const { nodes } = useGLTF("/models/EvilScreamingMan.glb") as unknown as GLTFResult;

  // Rendu filaire semi-transparent (comme ReferenceSphere) plutôt que la
  // texture bakée pleine : MeshBasicMaterial n'est pas affecté par
  // l'éclairage de la scène, donc la couleur reste vive et lisible.
  const material = useMemo(
    () =>
      new THREE.MeshBasicMaterial({
        color: HEAD_COLOR,
        wireframe: true,
        transparent: true,
        opacity: 0.55,
      }),
    [],
  );

  return (
    <group {...props} dispose={null}>
      <mesh geometry={nodes.EvilScreaminManBaseMesh.geometry} material={material} raycast={() => null} />
    </group>
  );
}

useGLTF.preload("/models/EvilScreamingMan.glb");
