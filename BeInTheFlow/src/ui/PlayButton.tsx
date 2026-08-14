import { useEffect, useRef, useState } from "react";
import { useSceneStore } from "../store/sceneStore";
import { useTrajectoryStore } from "../store/trajectoryStore";
import { renderScene } from "../utils/api";
import { connectAudioElement, resumeAudioContext } from "../utils/audioEngine";

export default function PlayButton() {
  const sources = useSceneStore((s) => s.sources);
  const trajectories = useTrajectoryStore((s) => s.trajectories);
  const [status, setStatus] = useState<"idle" | "rendering" | "error">("idle");
  const [error, setError] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const objectUrlRef = useRef<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);

  useEffect(() => {
    if (audioRef.current) connectAudioElement(audioRef.current);
  }, []);

  const handlePlay = async () => {
    // Résumé au tout début du handler : c'est ce clic qui constitue le geste
    // utilisateur débloquant l'AudioContext, qu'il y ait ou non des sources
    // prêtes — sinon un premier clic sans source ne déverrouille rien.
    resumeAudioContext();

    const ready = sources.filter((s) => s.path && !s.muted);
    if (ready.length === 0) {
      setStatus("error");
      setError("Aucune source active n'a de fichier audio assigné.");
      return;
    }

    setStatus("rendering");
    setError(null);
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      const blob = await renderScene({ sources: ready, trajectories }, controller.signal);

      if (objectUrlRef.current) {
        URL.revokeObjectURL(objectUrlRef.current);
      }
      const url = URL.createObjectURL(blob);
      objectUrlRef.current = url;
      setDownloadUrl(url);

      if (audioRef.current) {
        audioRef.current.src = url;
        await audioRef.current.play();
      }
      setStatus("idle");
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        setStatus("idle");
        setError(null);
      } else {
        setStatus("error");
        setError(err instanceof Error ? err.message : String(err));
      }
    } finally {
      abortRef.current = null;
    }
  };

  const handleStop = () => {
    abortRef.current?.abort();
  };

  return (
    <div className="play-panel">
      <div className="play-row">
        <button className="play-btn" onClick={handlePlay} disabled={status === "rendering"}>
          {status === "rendering" ? "Rendu en cours…" : "▶ Play"}
        </button>
        {status === "rendering" && (
          <button className="stop-btn" onClick={handleStop} aria-label="Stop" title="Stop">
            <span className="stop-icon">
              <span />
              <span />
            </span>
          </button>
        )}
      </div>
      {status === "error" && <p className="error">{error}</p>}
      <audio ref={audioRef} controls style={{ width: "100%", marginTop: "0.5rem" }} />
      {downloadUrl && (
        <a className="add-btn download-btn" href={downloadUrl} download="rendu_binaural.wav">
          ⬇ Télécharger le rendu
        </a>
      )}
    </div>
  );
}
