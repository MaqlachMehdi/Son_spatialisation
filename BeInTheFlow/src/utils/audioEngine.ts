// Pont Web Audio partagé entre PlayButton (qui possède l'élément <audio>) et
// WaveformVisualizer (qui a besoin de lire les échantillons en temps réel).
// Singleton hors state React : l'analyse audio est lue à 60fps par un rAF,
// pas par des re-renders.

export const FFT_SIZE = 1024;

let audioCtx: AudioContext | null = null;
let analyser: AnalyserNode | null = null;
let connectedElement: HTMLAudioElement | null = null;

// createMediaElementSource() ne peut être appelé qu'une seule fois par
// élément <audio> — d'où la garde sur connectedElement (utile en particulier
// avec le double-effet de React StrictMode en dev).
export function connectAudioElement(audio: HTMLAudioElement): AnalyserNode {
  if (connectedElement === audio && analyser) {
    return analyser;
  }

  audioCtx ??= new AudioContext();
  const source = audioCtx.createMediaElementSource(audio);
  analyser = audioCtx.createAnalyser();
  analyser.fftSize = FFT_SIZE;
  source.connect(analyser);
  analyser.connect(audioCtx.destination);
  connectedElement = audio;

  return analyser;
}

export function getAnalyser(): AnalyserNode | null {
  return analyser;
}

// Utilisé par TrajectoryPlayer pour synchroniser le déplacement des sources
// sur le temps de lecture réel de l'audio (currentTime/paused).
export function getAudioElement(): HTMLAudioElement | null {
  return connectedElement;
}

// Les navigateurs démarrent l'AudioContext suspendu tant qu'aucun geste
// utilisateur n'a eu lieu — à appeler depuis le handler de clic sur "Play".
export function resumeAudioContext(): void {
  void audioCtx?.resume();
}
