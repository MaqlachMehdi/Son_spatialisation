import type { TrajectoryDTO, TrajectoryType, TrajectoryWaypoint } from "../types";
import { useTrajectoryStore } from "../store/trajectoryStore";
import { useSceneStore } from "../store/sceneStore";

interface TrajectoryInspectorProps {
  trajectory: TrajectoryDTO;
}

const TYPE_LABELS: Record<TrajectoryType, string> = {
  circular: "Circulaire",
  elliptical: "Elliptique",
  points: "Nuage de points",
};

const emptyWaypoint: TrajectoryWaypoint = { azimuth: 0, elevation: 0, distance: 2.06 };

export default function TrajectoryInspector({ trajectory }: TrajectoryInspectorProps) {
  const updateTrajectory = useTrajectoryStore((s) => s.updateTrajectory);
  const applyToSelectedSource = useTrajectoryStore((s) => s.applyToSelectedSource);
  const playingTrajectoryId = useTrajectoryStore((s) => s.playingTrajectoryId);
  const playTrajectory = useTrajectoryStore((s) => s.playTrajectory);
  const stopTrajectory = useTrajectoryStore((s) => s.stopTrajectory);
  const sources = useSceneStore((s) => s.sources);
  const selectedSourceId = useSceneStore((s) => s.selectedId);
  const linkedSources = sources.filter((src) => src.trajectoryId === trajectory.id);

  const isPlaying = playingTrajectoryId === trajectory.id;

  const patch = (p: Partial<TrajectoryDTO>) => updateTrajectory(trajectory.id, p);

  const numberField = (
    key:
      | "distance"
      | "centerAzimuth"
      | "centerElevation"
      | "azAmplitude"
      | "elAmplitude"
      | "speed"
      | "axisAzimuth"
      | "axisElevation"
      | "offsetAzimuth"
      | "offsetElevation"
      | "offsetDistance",
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
        value={trajectory[key]}
        onChange={(e) => patch({ [key]: Number(e.target.value) })}
      />
    </label>
  );

  const updateWaypoint = (index: number, wp: Partial<TrajectoryWaypoint>) => {
    const points = trajectory.points.map((p, i) => (i === index ? { ...p, ...wp } : p));
    patch({ points });
  };

  const addWaypoint = () => patch({ points: [...trajectory.points, { ...emptyWaypoint }] });

  const removeWaypoint = (index: number) =>
    patch({ points: trajectory.points.filter((_, i) => i !== index) });

  return (
    <div className="inspector">
      <button
        className="add-btn"
        onClick={() => applyToSelectedSource(trajectory.id)}
        disabled={!selectedSourceId}
        title={selectedSourceId ? undefined : "Sélectionne d'abord une source dans le panneau des sources"}
      >
        Appliquer à la source sélectionnée
      </button>

      {!selectedSourceId ? (
        <p className="field-group-label">
          Sélectionne une source dans le panneau des sources (à droite) avant de cliquer sur « Appliquer ».
        </p>
      ) : linkedSources.length === 0 ? (
        <p className="field-group-label">
          Associe une source (bouton ci-dessus) pour pouvoir lire son déplacement.
        </p>
      ) : (
        <button
          className="play-btn"
          onClick={() => (isPlaying ? stopTrajectory() : playTrajectory(trajectory.id))}
        >
          {isPlaying ? "■ Arrêter la lecture" : "▶ Lire la trajectoire"}
        </button>
      )}

      <h2>Définir une trajectoire</h2>

      <label className="field">
        <span>Nom</span>
        <input
          type="text"
          value={trajectory.name}
          onChange={(e) => patch({ name: e.target.value })}
        />
      </label>

      <label className="field">
        <span>Type</span>
        <select
          value={trajectory.type}
          onChange={(e) => patch({ type: e.target.value as TrajectoryType })}
        >
          {(Object.keys(TYPE_LABELS) as TrajectoryType[]).map((t) => (
            <option key={t} value={t}>
              {TYPE_LABELS[t]}
            </option>
          ))}
        </select>
      </label>

      {numberField("speed", "Vitesse (×)", 0.1, 0.1, 5)}

      <button className="add-btn" onClick={() => patch({ reverse: !trajectory.reverse })}>
        {trajectory.reverse ? "⇄ Sens inversé" : "⇄ Inverser le sens"}
      </button>

      {(trajectory.type === "circular" || trajectory.type === "elliptical") && (
        <>
          {trajectory.type === "elliptical" && (
            <p className="field-group-label">Centre (glisser la poignée orange)</p>
          )}
          {numberField("distance", "Distance (m)", 0.1, 0.1)}
          {numberField("centerAzimuth", "Azimut centre (°)", 1, 0, 360)}
          {numberField("centerElevation", "Élévation centre (°)", 1, -90, 90)}
        </>
      )}

      {trajectory.type === "elliptical" && (
        <>
          <p className="field-group-label">
            Amplitude azimut (glisser la poignée bleue) / élévation (poignée verte)
          </p>
          {numberField("azAmplitude", "Amplitude azimut (°)", 1, 0, 180)}
          {numberField("elAmplitude", "Amplitude élévation (°)", 1, 0, 90)}
        </>
      )}

      {trajectory.type === "circular" && (
        <>
          <p className="field-group-label">Inclinaison (glisser la poignée bleue)</p>
          {numberField("axisAzimuth", "Azimut axe (°)", 1, 0, 360)}
          {numberField("axisElevation", "Élévation axe (°)", 1, -90, 90)}

          <p className="field-group-label">Décalage du centre (glisser la poignée orange)</p>
          {numberField("offsetAzimuth", "Azimut décalage (°)", 1, 0, 360)}
          {numberField("offsetElevation", "Élévation décalage (°)", 1, -90, 90)}
          {numberField("offsetDistance", "Distance décalage (m)", 0.1, 0)}
        </>
      )}

      {trajectory.type === "points" && (
        <div className="waypoint-list">
          <p className="field-group-label">
            Chaque point est aussi déplaçable directement dans la scène (poignée orange).
          </p>
          {trajectory.points.map((p, i) => (
            <div key={i} className="waypoint-row">
              <input
                type="number"
                step={1}
                value={p.azimuth}
                onChange={(e) => updateWaypoint(i, { azimuth: Number(e.target.value) })}
                title="Azimut (°)"
              />
              <input
                type="number"
                step={1}
                value={p.elevation}
                onChange={(e) => updateWaypoint(i, { elevation: Number(e.target.value) })}
                title="Élévation (°)"
              />
              <input
                type="number"
                step={0.1}
                value={p.distance}
                onChange={(e) => updateWaypoint(i, { distance: Number(e.target.value) })}
                title="Distance (m)"
              />
              <button
                className="remove-btn"
                onClick={() => removeWaypoint(i)}
                aria-label={`Supprimer le point ${i + 1}`}
              >
                ×
              </button>
            </div>
          ))}
          <button className="add-btn" onClick={addWaypoint}>
            + Ajouter un point
          </button>
        </div>
      )}
    </div>
  );
}
