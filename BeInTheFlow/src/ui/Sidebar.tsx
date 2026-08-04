import { useState } from "react";
import { useSceneStore } from "../store/sceneStore";
import SourceInspector from "./SourceInspector";

export default function Sidebar() {
  const sources = useSceneStore((s) => s.sources);
  const selectedId = useSceneStore((s) => s.selectedId);
  const selectSource = useSceneStore((s) => s.selectSource);
  const addSource = useSceneStore((s) => s.addSource);
  const removeSource = useSceneStore((s) => s.removeSource);
  const [collapsed, setCollapsed] = useState(false);

  const selected = sources.find((s) => s.id === selectedId) ?? null;

  return (
    <div className="sidebar-dock">
      <button
        className="sidebar-toggle"
        onClick={() => setCollapsed((c) => !c)}
        aria-label={collapsed ? "Ouvrir le panneau" : "Replier le panneau"}
        aria-expanded={!collapsed}
      >
        <span />
        <span />
        <span />
      </button>

      <div className={`sidebar ${collapsed ? "collapsed" : ""}`}>
        <div className="sidebar-inner">
          <div className="sidebar-header">
            <h1>Sources</h1>
          </div>
          <div className="source-list">
            {sources.map((source) => (
              <div
                key={source.id}
                className={`source-row ${source.id === selectedId ? "selected" : ""}`}
                onClick={() => selectSource(source.id)}
              >
                <span className="source-swatch" style={{ backgroundColor: source.color }} />
                <span className="source-name">{source.label}</span>
                <button
                  className="remove-btn"
                  onClick={(e) => {
                    e.stopPropagation();
                    removeSource(source.id);
                  }}
                  aria-label={`Supprimer ${source.label}`}
                >
                  ×
                </button>
              </div>
            ))}
            <button className="add-btn" onClick={addSource}>
              + Ajouter une source
            </button>
          </div>

          {selected && <SourceInspector source={selected} />}
        </div>
      </div>
    </div>
  );
}
