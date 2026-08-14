import { useState } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Grid } from "@react-three/drei";
import { registerOrbitControls } from "../utils/cameraControl";
import ListenerHead from "./ListenerHead";
import ReferenceSphere from "./ReferenceSphere";
import SourceNode from "./SourceNode";
import TrajectoryLine from "./TrajectoryLine";
import TrajectoryHandles from "./TrajectoryHandles";
import EllipseHandles from "./EllipseHandles";
import PointsHandles from "./PointsHandles";
import TrajectoryPlayer from "./TrajectoryPlayer";
import { useSceneStore } from "../store/sceneStore";
import { useTrajectoryStore } from "../store/trajectoryStore";

const REFERENCE_RADIUS = 2.06; // distance IRCAM LISTEN, cf. src/scene/Soundsource.py

export default function SceneCanvas() {
  const sources = useSceneStore((s) => s.sources);
  const selectedId = useSceneStore((s) => s.selectedId);
  const selectSource = useSceneStore((s) => s.selectSource);
  const trajectories = useTrajectoryStore((s) => s.trajectories);
  const selectedTrajectoryId = useTrajectoryStore((s) => s.selectedTrajectoryId);
  const [isDragging, setIsDragging] = useState(false);
  const selectedTrajectory = trajectories.find((t) => t.id === selectedTrajectoryId) ?? null;

  return (
    <Canvas camera={{ position: [0, 0, -5], fov: 50 }}>
      <ambientLight intensity={0.6} />
      <pointLight position={[5, 5, 5]} intensity={0.8} />
      <pointLight position={[-5, 3, -5]} intensity={0.4} />

      <OrbitControls ref={registerOrbitControls} target={[0, 0, 0]} enabled={!isDragging} makeDefault />
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
      <TrajectoryPlayer />
      {selectedTrajectory && <TrajectoryLine trajectory={selectedTrajectory} />}
      {selectedTrajectory && selectedTrajectory.type === "circular" && (
        <TrajectoryHandles trajectory={selectedTrajectory} onDragStateChange={setIsDragging} />
      )}
      {selectedTrajectory && selectedTrajectory.type === "elliptical" && (
        <EllipseHandles trajectory={selectedTrajectory} onDragStateChange={setIsDragging} />
      )}
      {selectedTrajectory && selectedTrajectory.type === "points" && (
        <PointsHandles trajectory={selectedTrajectory} onDragStateChange={setIsDragging} />
      )}

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