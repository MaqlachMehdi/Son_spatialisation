import { useState } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Grid } from "@react-three/drei";
import ListenerHead from "./ListenerHead";
import ReferenceSphere from "./ReferenceSphere";
import SourceNode from "./SourceNode";
import { useSceneStore } from "../store/sceneStore";

const REFERENCE_RADIUS = 2.06; // distance IRCAM LISTEN, cf. src/scene/Soundsource.py

export default function SceneCanvas() {
  const sources = useSceneStore((s) => s.sources);
  const selectedId = useSceneStore((s) => s.selectedId);
  const selectSource = useSceneStore((s) => s.selectSource);
  const [isDragging, setIsDragging] = useState(false);

  return (
    <Canvas camera={{ position: [0, 2.5, 5], fov: 50 }}>
      <ambientLight intensity={0.6} />
      <pointLight position={[5, 5, 5]} intensity={0.8} />
      <pointLight position={[-5, 3, -5]} intensity={0.4} />

      <OrbitControls enabled={!isDragging} makeDefault />
      <Grid
        args={[10, 10]}
        position={[0, -1.2, 0]}
        cellColor="#3a3a3d"
        sectionColor="#6f2232"
        fadeDistance={12}
        raycast={() => null}
      />

      <ListenerHead />
      <ReferenceSphere radius={REFERENCE_RADIUS} />

      {sources.map((source) => (
        <SourceNode
          key={source.id}
          source={source}
          selected={source.id === selectedId}
          onSelect={() => selectSource(source.id)}
          onDragStateChange={setIsDragging}
        />
      ))}
    </Canvas>
  );
}