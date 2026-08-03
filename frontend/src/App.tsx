import SceneCanvas from "./three/SceneCanvas";
import Sidebar from "./ui/Sidebar";
import "./App.css";

function App() {
  return (
    <div className="app">
      <div className="canvas-wrap">
        <SceneCanvas />
      </div>
      <Sidebar />
    </div>
  );
}

export default App;