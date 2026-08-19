import { useEffect, useRef, useState } from "react";
import { useAuthStore } from "../store/authStore";
import { useSoundsStore } from "../store/soundsStore";
import { uploadSound } from "../utils/api";

const MAX_DURATION_SEC = 120;

// Ordre de préférence : webm/opus est le plus largement supporté (Chrome,
// Edge, Firefox), mp4/aac sert de repli pour Safari qui ne supporte pas
// webm en enregistrement. Le format compressé exact n'a pas d'importance :
// le backend le réencode de toute façon en wav (voir routers/sounds.py).
const MIME_CANDIDATES = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4"];

function pickMimeType(): string {
  if (typeof MediaRecorder === "undefined") return "";
  return MIME_CANDIDATES.find((t) => MediaRecorder.isTypeSupported(t)) ?? "";
}

function extensionFor(mimeType: string): string {
  if (mimeType.includes("mp4")) return "m4a";
  if (mimeType.includes("ogg")) return "ogg";
  return "webm";
}

function formatTime(totalSeconds: number): string {
  const m = Math.floor(totalSeconds / 60);
  const s = totalSeconds % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

function defaultLabel(): string {
  const now = new Date();
  const date = now.toLocaleDateString("fr-FR");
  const time = now.toLocaleTimeString("fr-FR", { hour: "2-digit", minute: "2-digit" });
  return `Enregistrement du ${date} ${time}`;
}

type Phase = "idle" | "recording" | "preview" | "uploading";

export default function RecordSoundButton() {
  const isAuthenticated = useAuthStore((s) => s.status === "authenticated");
  const refreshSounds = useSoundsStore((s) => s.refreshSounds);

  const [phase, setPhase] = useState<Phase>("idle");
  const [elapsedSec, setElapsedSec] = useState(0);
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [label, setLabel] = useState("");
  const [error, setError] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const autoStopRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Libère le micro et les timers si le panneau se ferme/démonte pendant un
  // enregistrement en cours — sinon le témoin "micro actif" du navigateur
  // resterait allumé indéfiniment.
  useEffect(() => {
    return () => {
      streamRef.current?.getTracks().forEach((t) => t.stop());
      if (timerRef.current) clearInterval(timerRef.current);
      if (autoStopRef.current) clearTimeout(autoStopRef.current);
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const stopRecording = () => {
    mediaRecorderRef.current?.stop();
    if (timerRef.current) clearInterval(timerRef.current);
    if (autoStopRef.current) clearTimeout(autoStopRef.current);
  };

  const startRecording = async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const mimeType = pickMimeType();
      const recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
      mediaRecorderRef.current = recorder;
      chunksRef.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: recorder.mimeType || mimeType || "audio/webm" });
        setRecordedBlob(blob);
        setPreviewUrl(URL.createObjectURL(blob));
        setLabel(defaultLabel());
        setPhase("preview");
        streamRef.current?.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
      };

      recorder.start();
      setPhase("recording");
      setElapsedSec(0);
      timerRef.current = setInterval(() => setElapsedSec((s) => s + 1), 1000);
      autoStopRef.current = setTimeout(stopRecording, MAX_DURATION_SEC * 1000);
    } catch (err) {
      setError(
        err instanceof Error
          ? `Accès au micro refusé ou indisponible (${err.message}).`
          : "Accès au micro refusé ou indisponible.",
      );
    }
  };

  const handleRetry = () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setRecordedBlob(null);
    setPreviewUrl(null);
    setError(null);
    setPhase("idle");
  };

  const handleConfirmUpload = async () => {
    if (!recordedBlob) return;
    setPhase("uploading");
    setError(null);
    try {
      const ext = extensionFor(recordedBlob.type);
      const filename = `${label.trim() || defaultLabel()}.${ext}`;
      const file = new File([recordedBlob], filename, { type: recordedBlob.type });
      await uploadSound(file);
      await refreshSounds();
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      setRecordedBlob(null);
      setPreviewUrl(null);
      setPhase("idle");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setPhase("preview");
    }
  };

  if (phase === "recording") {
    return (
      <div className="record-panel">
        <div className="recording-indicator">
          <span className="recording-dot" />
          Enregistrement… {formatTime(elapsedSec)} / {formatTime(MAX_DURATION_SEC)}
        </div>
        <button className="stop-btn record-stop-btn" onClick={stopRecording}>
          ■ Arrêter
        </button>
      </div>
    );
  }

  if (phase === "preview" || phase === "uploading") {
    return (
      <div className="record-panel">
        {previewUrl && <audio controls src={previewUrl} style={{ width: "100%" }} />}
        <label className="field">
          <span>Nom du son</span>
          <input
            type="text"
            value={label}
            onChange={(e) => setLabel(e.target.value)}
            disabled={phase === "uploading"}
          />
        </label>
        {error && <p className="error">{error}</p>}
        <div className="record-actions">
          <button className="add-btn" onClick={handleRetry} disabled={phase === "uploading"}>
            Recommencer
          </button>
          <button className="play-btn" onClick={handleConfirmUpload} disabled={phase === "uploading"}>
            {phase === "uploading" ? "Envoi…" : "Enregistrer"}
          </button>
        </div>
      </div>
    );
  }

  return (
    <>
      <button
        className="add-btn"
        onClick={startRecording}
        disabled={!isAuthenticated}
        title={isAuthenticated ? undefined : "Connecte-toi pour enregistrer un son"}
      >
        ● Enregistrer un son
      </button>
      {error && <p className="error">{error}</p>}
    </>
  );
}
