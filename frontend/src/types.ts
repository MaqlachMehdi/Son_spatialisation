// Champs alignés sur SoundSource (src/scene/Soundsource.py)
export interface SoundSourceDTO {
  id: string;
  label: string;
  path: string;
  azimuth: number;   // degrés, convention SOFA : 0°=devant, 90°=gauche
  elevation: number; // degrés, -90°=bas, 90°=dessus
  distance: number;  // mètres
  gain: number;      // gain linéaire
  color: string;
}

export interface SoundAsset {
  id: string;
  label: string;
  path: string;
}

export interface RenderRequest {
  sources: {
    path: string;
    azimuth: number;
    elevation: number;
    distance: number;
    gain: number;
    label: string;
  }[];
}