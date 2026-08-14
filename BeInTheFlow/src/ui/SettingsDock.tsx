import { useEffect, useState } from "react";
import { useSettingsStore } from "../store/settingsStore";
import { fetchHrtfs } from "../utils/api";
import { resetCamera } from "../utils/cameraControl";
import type { HrtfAsset } from "../types";

// Roue de réglage classique (icône générique "settings").
function GearIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
    </svg>
  );
}

function CameraIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
      <circle cx="12" cy="13" r="4" />
    </svg>
  );
}

// Recentrer/réinitialiser la vue caméra (icône "crosshair" classique).
function RecenterIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="9" />
      <line x1="22" y1="12" x2="18" y2="12" />
      <line x1="6" y1="12" x2="2" y2="12" />
      <line x1="12" y1="6" x2="12" y2="2" />
      <line x1="12" y1="22" x2="12" y2="18" />
    </svg>
  );
}

export default function SettingsDock() {
  const [collapsed, setCollapsed] = useState(true);
  const [hrtfs, setHrtfs] = useState<HrtfAsset[]>([]);
  const [hrtfError, setHrtfError] = useState<string | null>(null);
  const selectedHrtfId = useSettingsStore((s) => s.selectedHrtfId);
  const setSelectedHrtf = useSettingsStore((s) => s.setSelectedHrtf);

  useEffect(() => {
    fetchHrtfs()
      .then((list) => {
        setHrtfs(list);
        if (!selectedHrtfId) {
          const active = list.find((h) => h.active) ?? list[0];
          if (active) setSelectedHrtf(active.id);
        }
      })
      .catch((err: Error) => setHrtfError(err.message));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="settings-dock">
      <button
        className="sidebar-toggle settings-toggle"
        onClick={() => setCollapsed((c) => !c)}
        aria-label={collapsed ? "Ouvrir les réglages" : "Replier les réglages"}
        aria-expanded={!collapsed}
      >
        <GearIcon />
      </button>

      <button
        className="sidebar-toggle settings-toggle"
        onClick={resetCamera}
        aria-label="Recentrer la caméra"
        title="Recentrer la caméra"
      >
        <RecenterIcon />
      </button>

      <div className={`settings-panel ${collapsed ? "collapsed" : ""}`}>
        <div className="settings-panel-inner">
          <div className="sidebar-header">
            <h1>Réglages</h1>
          </div>

          <label className="field">
            <span>HRTF sélectionnée</span>
            {hrtfError ? (
              <span className="error">Backend indisponible ({hrtfError})</span>
            ) : (
              <select value={selectedHrtfId ?? ""} onChange={(e) => setSelectedHrtf(e.target.value)}>
                {hrtfs.length === 0 && <option value="">— chargement —</option>}
                {hrtfs.map((h) => (
                  <option key={h.id} value={h.id}>
                    {h.label}
                  </option>
                ))}
              </select>
            )}
          </label>

          <button className="add-btn settings-action" onClick={() => {}}>
            <CameraIcon />
            Trouver sa HRTF optimale
          </button>

          <button className="add-btn" onClick={() => {}}>
            Importer un son
          </button>

          <button className="add-btn" onClick={() => {}}>
            Enregistrer un son
          </button>
        </div>
      </div>
    </div>
  );
}
