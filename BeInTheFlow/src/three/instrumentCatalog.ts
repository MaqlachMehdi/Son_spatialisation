// Catalogue des modèles 3D pouvant remplacer le cercle d'une source.
// Fichiers sources dans src/3D_design/Instruments/, copiés en /public/models/instruments/
// pour être chargeables par URL (voir EvilScreamingMan.glb pour le même principe).
export type InstrumentFormat = "fbx" | "3ds";

export interface InstrumentCatalogEntry {
  id: string;
  label: string;
  url: string;
  format: InstrumentFormat;
  scale: number;
  // Rotation additionnelle (radians) pour compenser l'axe "avant" propre à
  // chaque modèle, appliquée après l'orientation vers le centre. À ajuster
  // au cas par cas selon comment chaque asset a été modélisé.
  rotationOffset: [number, number, number];
}

export const INSTRUMENT_CATALOG: InstrumentCatalogEntry[] = [
  {
    id: "bugle",
    label: "Bugle",
    url: "/models/instruments/Bugle.fbx",
    format: "fbx",
    scale: 0.01,
    // -90° en Y pour que le pavillon pointe vers le centre une fois
    // l'orientation "lookAt" appliquée (le +90° initial pointait à l'opposé).
    rotationOffset: [0, -Math.PI / 2, 0],
  },
  {
    id: "oud",
    label: "Oud",
    url: "/models/instruments/Oud.3ds",
    format: "3ds",
    scale: 0.01,
    rotationOffset: [0, 0, 0],
  },
];

// Piano.blend n'est pas inclus : le format Blender natif n'a pas de loader
// three.js, il faudrait l'exporter en glTF/FBX depuis Blender avant de
// pouvoir l'ajouter au catalogue ci-dessus.
export const UNAVAILABLE_INSTRUMENTS = [{ id: "piano", label: "Piano (à convertir en glTF/FBX)" }];

export function getInstrument(id: string | null | undefined): InstrumentCatalogEntry | null {
  if (!id) return null;
  return INSTRUMENT_CATALOG.find((m) => m.id === id) ?? null;
}
