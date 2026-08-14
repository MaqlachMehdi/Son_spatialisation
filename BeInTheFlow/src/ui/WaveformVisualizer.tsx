import { useEffect, useRef } from "react";
import { FFT_SIZE, getAnalyser } from "../utils/audioEngine";

const GLOW_COLOR = "#8b0000"; // dark_red — halo
const CORE_COLOR = "#ff2b4d"; // cœur du trait, plus vif pour lire sur le halo

// Oscilloscope temps réel : lit l'AnalyserNode partagé (audioEngine) à
// chaque frame et trace la forme d'onde avec un glow (double passe :
// halo flou large + trait net par-dessus). Rendu en Canvas 2D, indépendant
// du Canvas Three.js de la scène.
export default function WaveformVisualizer() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const resize = () => {
      const { width, height } = canvas.getBoundingClientRect();
      canvas.width = width * dpr;
      canvas.height = height * dpr;
    };
    resize();
    window.addEventListener("resize", resize);

    const data = new Uint8Array(FFT_SIZE);
    let frameId: number;

    const drawPass = (blur: number, color: string, lineWidth: number, alpha: number) => {
      const { width, height } = canvas;
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.shadowBlur = blur;
      ctx.shadowColor = color;
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      ctx.beginPath();
      const sliceWidth = width / FFT_SIZE;
      let x = 0;
      for (let i = 0; i < FFT_SIZE; i++) {
        const v = data[i] / 128 - 1; // -1..1
        const y = height / 2 + v * height * 0.42;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
        x += sliceWidth;
      }
      ctx.stroke();
      ctx.restore();
    };

    const draw = () => {
      frameId = requestAnimationFrame(draw);
      const analyser = getAnalyser();
      if (analyser) {
        analyser.getByteTimeDomainData(data);
      } else {
        data.fill(128); // silence = ligne plate, tant qu'aucun son n'a encore été lancé
      }

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      drawPass(28, GLOW_COLOR, 6, 0.55); // halo
      drawPass(8, CORE_COLOR, 1.5, 1); // cœur
    };
    draw();

    return () => {
      cancelAnimationFrame(frameId);
      window.removeEventListener("resize", resize);
    };
  }, []);

  return (
    <div className="waveform-dock">
      <div className="waveform-box">
        <canvas ref={canvasRef} className="waveform-canvas" />
      </div>
    </div>
  );
}
