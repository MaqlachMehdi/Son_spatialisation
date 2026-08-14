// Pont entre les OrbitControls (montés dans le <Canvas> Three.js) et les
// boutons React classiques hors du canvas — même principe que audioEngine.ts.
interface ResettableControls {
  reset: () => void;
}

let controls: ResettableControls | null = null;

export function registerOrbitControls(instance: ResettableControls | null): void {
  controls = instance;
}

export function resetCamera(): void {
  controls?.reset();
}
