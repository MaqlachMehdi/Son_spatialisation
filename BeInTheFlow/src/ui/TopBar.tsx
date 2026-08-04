import PlayButton from "./PlayButton";

// Barre persistante en haut à droite : titre + lecture, toujours visible,
// indépendante des panneaux déroulants (sources / trajectoires).
export default function TopBar() {
  return (
    <div className="top-bar">
      <h1>Spatialisation HRTF</h1>
      <PlayButton />
    </div>
  );
}
