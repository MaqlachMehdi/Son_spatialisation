import SceneCanvas from "./three/SceneCanvas";
import Sidebar from "./ui/Sidebar";
import TrajectoryDock from "./ui/TrajectoryDock";
import TopBar from "./ui/TopBar";
import "./App.css";

function App() {
  return (
    <div className="app">
      <div className="canvas-wrap">
        <SceneCanvas />
      </div>
      <TopBar />
      <TrajectoryDock />
      <Sidebar />
    </div>
  );
}

export default App;