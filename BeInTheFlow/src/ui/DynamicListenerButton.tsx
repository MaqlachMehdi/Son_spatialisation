import { useRef, useState } from "react";
import { useListenerStore } from "../store/listenerStore";
import { startListenerCapture, type ListenerCaptureHandle } from "../utils/phoneMotion";

// Silhouette qui marche — même gabarit que les icônes de SettingsDock.tsx
// (trait, viewBox 24x24, strokeWidth 1.8).
function WalkIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="4" r="2" />
      <path d="M12 6 L12 13" />
      <path d="M12 9 L7 12" />
      <path d="M12 9 L17 11" />
      <path d="M12 13 L8 21" />
      <path d="M12 13 L15 20" />
    </svg>
  );
}

// Bouton icône unique, deux états (démarrer / arrêter capture du mouvement du
// téléphone) — même style que le bouton "recentrer" de SettingsDock.tsx,
// passe au rouge pendant l'enregistrement d'une trajectoire.
export default function DynamicListenerButton() {
  const dynamicMode = useListenerStore((s) => s.dynamicMode);
  const startCapture = useListenerStore((s) => s.startCapture);
  const stopCapture = useListenerStore((s) => s.stopCapture);
  const updatePose = useListenerStore((s) => s.updatePose);
  const recordWaypoint = useListenerStore((s) => s.recordWaypoint);

  const [error, setError] = useState<string | null>(null);
  const handleRef = useRef<ListenerCaptureHandle | null>(null);

  const handleStart = async () => {
    setError(null);
    try {
      // La demande de permission iOS (dans startListenerCapture) doit être
      // déclenchée par ce clic — un appel différé (ex. dans un useEffect) est
      // silencieusement refusé par Safari.
      handleRef.current = await startListenerCapture(updatePose, recordWaypoint);
      startCapture();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const handleStop = () => {
    handleRef.current?.stop();
    handleRef.current = null;
    stopCapture();
  };

  const label = dynamicMode ? "Arrêter l'auditeur dynamique" : "Démarrer l'auditeur dynamique";

  return (
    <div className="listener-toggle-wrap">
      <button
        className={`sidebar-toggle listener-toggle ${dynamicMode ? "recording" : ""}`}
        onClick={dynamicMode ? handleStop : handleStart}
        aria-label={label}
        aria-pressed={dynamicMode}
        title={error ?? label}
      >
        <WalkIcon />
      </button>
      {error && <p className="error listener-error">{error}</p>}
    </div>
  );
}
