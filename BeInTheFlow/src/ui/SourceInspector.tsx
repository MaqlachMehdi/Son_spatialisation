import { useEffect, useState } from "react";
import type { SoundSourceDTO, SoundAsset } from "../types";
import { useSceneStore } from "../store/sceneStore";
import { fetchSounds } from "../utils/api";
import { INSTRUMENT_CATALOG, UNAVAILABLE_INSTRUMENTS } from "../three/instrumentCatalog";

interface SourceInspectorProps {
  source: SoundSourceDTO;
}

export default function SourceInspector({ source }: SourceInspectorProps) {
  const updateSource = useSceneStore((s) => s.updateSource);
  const [sounds, setSounds] = useState<SoundAsset[]>([]);
  const [soundsError, setSoundsError] = useState<string | null>(null);

  useEffect(() => {
    fetchSounds()
      .then(setSounds)
      .catch((err: Error) => setSoundsError(err.message));
  }, []);

  const field = (
    key: "azimuth" | "elevation" | "distance" | "gain",
    label: string,
    step: number,
    min?: number,
    max?: number,
  ) => (
    <label className="field">
      <span>{label}</span>
      <input
        type="number"
        step={step}
        min={min}
        max={max}
        value={source[key]}
        onChange={(e) => updateSource(source.id, { [key]: Number(e.target.value) })}
      />
    </label>
  );

  return (
    <div className="inspector">
      <h2>Source sélectionnée</h2>

      <label className="field">
        <span>Nom</span>
        <input
          type="text"
          value={source.label}
          onChange={(e) => updateSource(source.id, { label: e.target.value })}
        />
      </label>

      <label className="field">
        <span>Fichier audio</span>
        {soundsError ? (
          <span className="error">Backend indisponible ({soundsError})</span>
        ) : (
          <select
            value={source.path}
            onChange={(e) => updateSource(source.id, { path: e.target.value })}
          >
            <option value="">— choisir —</option>
            {sounds.map((s) => (
              <option key={s.id} value={s.path}>
                {s.label}
              </option>
            ))}
          </select>
        )}
      </label>

      <label className="field">
        <span>Modèle 3D</span>
        <select
          value={source.modelId ?? ""}
          onChange={(e) => updateSource(source.id, { modelId: e.target.value || null })}
        >
          <option value="">Cercle (par défaut)</option>
          {INSTRUMENT_CATALOG.map((m) => (
            <option key={m.id} value={m.id}>
              {m.label}
            </option>
          ))}
          {UNAVAILABLE_INSTRUMENTS.map((m) => (
            <option key={m.id} value="" disabled>
              {m.label}
            </option>
          ))}
        </select>
      </label>

      {field("azimuth", "Azimut (°)", 1, 0, 360)}
      {field("elevation", "Élévation (°)", 1, -90, 90)}
      {field("distance", "Distance (m)", 0.1, 0.1)}
      {field("gain", "Gain", 0.05, 0)}
    </div>
  );
}