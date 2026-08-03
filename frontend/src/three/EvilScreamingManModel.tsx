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

export function Model(props: ThreeElements["group"]) {
  const { nodes, materials } = useGLTF("/models/EvilScreamingMan.glb") as unknown as GLTFResult;
  return (
    <group {...props} dispose={null}>
      <mesh geometry={nodes.EvilScreaminManBaseMesh.geometry} material={materials.EvilScrMan} />
    </group>
  );
}

useGLTF.preload("/models/EvilScreamingMan.glb");
