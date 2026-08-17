import PlayButton from "./PlayButton";

// Barre persistante en haut à droite : titre + lecture, toujours visible,
// indépendante des panneaux déroulants (sources / trajectoires).
export default function TopBar() {
  return (
    <div className="top-bar">
      <div className="top-bar-header">
        <img src="/logo.png" alt="BeInTheFlow" className="top-bar-logo" />
        <h1>Spatialisation HRTF</h1>
      </div>
      <PlayButton />
    </div>
  );
}
