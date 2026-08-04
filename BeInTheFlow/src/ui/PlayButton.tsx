import { useRef, useState } from "react";
import { useSceneStore } from "../store/sceneStore";
import { renderScene } from "../utils/api";

export default function PlayButton() {
  const sources = useSceneStore((s) => s.sources);
  const [status, setStatus] = useState<"idle" | "rendering" | "error">("idle");
  const [error, setError] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const objectUrlRef = useRef<string | null>(null);

  const handlePlay = async () => {
    const ready = sources.filter((s) => s.path);
    if (ready.length === 0) {
      setStatus("error");
      setError("Aucune source n'a de fichier audio assigné.");
      return;
    }

    setStatus("rendering");
    setError(null);
    try {
      const blob = await renderScene({
        sources: ready.map((s) => ({
          path: s.path,
          azimuth: s.azimuth,
          elevation: s.elevation,
          distance: s.distance,
          gain: s.gain,
          label: s.label,
        })),
      });

      if (objectUrlRef.current) {
        URL.revokeObjectURL(objectUrlRef.current);
      }
      const url = URL.createObjectURL(blob);
      objectUrlRef.current = url;

      if (audioRef.current) {
        audioRef.current.src = url;
        await audioRef.current.play();
      }
      setStatus("idle");
    } catch (err) {
      setStatus("error");
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  return (
    <div className="play-panel">
      <button className="play-btn" onClick={handlePlay} disabled={status === "rendering"}>
        {status === "rendering" ? "Rendu en cours…" : "▶ Play"}
      </button>
      {status === "error" && <p className="error">{error}</p>}
      <audio ref={audioRef} controls style={{ width: "100%", marginTop: "0.5rem" }} />
    </div>
  );
}