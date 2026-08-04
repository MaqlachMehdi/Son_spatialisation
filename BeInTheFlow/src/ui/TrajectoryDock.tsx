import { useState } from "react";
import { useTrajectoryStore } from "../store/trajectoryStore";
import TrajectoryInspector from "./TrajectoryInspector";

// Icône : flèche en pointillé décrivant un S allongé — évoque une trajectoire
// libre, par opposition au hamburger du panneau des sources.
function DashedSArrowIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
      <path
        d="M6 22 C6 15, 18 16, 18 9 C18 5, 12 3, 9 2"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeDasharray="3 3"
      />
      <path
        d="M5.5 4.5 L9 2 L9.5 6"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
        fill="none"
      />
    </svg>
  );
}

export default function TrajectoryDock() {
  const trajectories = useTrajectoryStore((s) => s.trajectories);
  const selectedTrajectoryId = useTrajectoryStore((s) => s.selectedTrajectoryId);
  const selectTrajectory = useTrajectoryStore((s) => s.selectTrajectory);
  const addTrajectory = useTrajectoryStore((s) => s.addTrajectory);
  const removeTrajectory = useTrajectoryStore((s) => s.removeTrajectory);
  const [collapsed, setCollapsed] = useState(true);

  const selected = trajectories.find((t) => t.id === selectedTrajectoryId) ?? null;

  return (
    <div className="sidebar-dock dock-left">
      <button
        className="sidebar-toggle trajectory-toggle"
        onClick={() => setCollapsed((c) => !c)}
        aria-label={collapsed ? "Ouvrir les trajectoires" : "Replier les trajectoires"}
        aria-expanded={!collapsed}
      >
        <DashedSArrowIcon />
      </button>

      <div className={`sidebar ${collapsed ? "collapsed" : ""}`}>
        <div className="sidebar-inner">
          <div className="sidebar-header">
            <h1>Trajectoires</h1>
          </div>

          <div className="source-list">
            {trajectories.map((trajectory) => (
              <div
                key={trajectory.id}
                className={`source-row ${trajectory.id === selectedTrajectoryId ? "selected" : ""}`}
                onClick={() => selectTrajectory(trajectory.id)}
              >
                <span className="source-name">{trajectory.name}</span>
                <button
                  className="remove-btn"
                  onClick={(e) => {
                    e.stopPropagation();
                    removeTrajectory(trajectory.id);
                  }}
                  aria-label={`Supprimer ${trajectory.name}`}
                >
                  ×
                </button>
              </div>
            ))}
            <button className="add-btn" onClick={() => addTrajectory()}>
              + Ajouter une trajectoire
            </button>
          </div>

          {selected && <TrajectoryInspector trajectory={selected} />}
        </div>
      </div>
    </div>
  );
}
