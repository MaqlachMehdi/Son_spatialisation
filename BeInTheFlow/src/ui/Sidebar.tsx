import { useState } from "react";
import { useSceneStore } from "../store/sceneStore";
import SourceInspector from "./SourceInspector";

function SpeakerIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
      <path d="M15.54 8.46a5 5 0 0 1 0 7.07" />
      <path d="M19.07 4.93a10 10 0 0 1 0 14.14" />
    </svg>
  );
}

function SpeakerMutedIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
      <line x1="23" y1="9" x2="17" y2="15" />
      <line x1="17" y1="9" x2="23" y2="15" />
    </svg>
  );
}

export default function Sidebar() {
  const sources = useSceneStore((s) => s.sources);
  const selectedId = useSceneStore((s) => s.selectedId);
  const selectSource = useSceneStore((s) => s.selectSource);
  const addSource = useSceneStore((s) => s.addSource);
  const removeSource = useSceneStore((s) => s.removeSource);
  const updateSource = useSceneStore((s) => s.updateSource);
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
                  className="mute-btn"
                  onClick={(e) => {
                    e.stopPropagation();
                    updateSource(source.id, { muted: !source.muted });
                  }}
                  aria-label={source.muted ? `Réactiver ${source.label}` : `Couper ${source.label}`}
                  aria-pressed={source.muted}
                >
                  {source.muted ? <SpeakerMutedIcon /> : <SpeakerIcon />}
                </button>
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
