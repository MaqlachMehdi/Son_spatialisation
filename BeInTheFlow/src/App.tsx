import { useEffect } from "react";
import SceneCanvas from "./three/SceneCanvas";
import Sidebar from "./ui/Sidebar";
import TrajectoryDock from "./ui/TrajectoryDock";
import TopBar from "./ui/TopBar";
import SettingsDock from "./ui/SettingsDock";
import AccountDock from "./ui/AccountDock";
import WaveformVisualizer from "./ui/WaveformVisualizer";
import { useAuthStore } from "./store/authStore";
import "./App.css";

function App() {
  const checkSession = useAuthStore((s) => s.checkSession);

  useEffect(() => {
    checkSession();
  }, [checkSession]);

  return (
    <div className="app">
      <div className="canvas-wrap">
        <SceneCanvas />
      </div>
      <TopBar />
      <SettingsDock />
      <AccountDock />
      <TrajectoryDock />
      <Sidebar />
      <WaveformVisualizer />
    </div>
  );
}

export default App;