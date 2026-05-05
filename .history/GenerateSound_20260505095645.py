import numpy as np
import soundfile as sf
from pathlib import Path

SAVE_DIR = Path(__file__).parent / "sound" / "generated"

_GAIN = 1 / np.sqrt(100_000)  # -50 dBFS, évite la saturation


class Impulse:
    """Impulsion de Dirac répétée à une fréquence de répétition donnée.

    Paramètres
    ----------
    frequency   : nombre d'impulsions par seconde (Hz de répétition)
    duration    : durée totale du signal en secondes
    sr          : taux d'échantillonnage
    amplitude   : amplitude de chaque impulsion (0.0 – 1.0)
    save        : si True, sauvegarde dans sound/generated/
    """

    def __init__(
        self,
        frequency: float = 1.0,
        duration: float = 10.0,
        sr: int = 44_100,
        amplitude: float = 1.0,
        save: bool = False,
    ):
        if frequency <= 0:
            raise ValueError("frequency doit être > 0")
        self.frequency = frequency
        self.duration = duration
        self.sr = sr
        self.amplitude = amplitude
        self.save = save
        self._signal: np.ndarray | None = None

    def generate(self) -> np.ndarray:
        n_samples = int(self.sr * self.duration)
        signal = np.zeros(n_samples, dtype=np.float32)

        period = int(self.sr / self.frequency)
        for i in range(n_samples // period):
            signal[i * period] = self.amplitude

        self._signal = signal

        if self.save:
            self._write(f"impulse_{self.frequency}Hz_{self.duration}s.wav")

        return signal

    def _write(self, filename: str) -> None:
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        path = SAVE_DIR / filename
        sf.write(str(path), self._signal, self.sr)
        print(f"[Impulse] Sauvegardé : {path}")


class Note:
    """Signal sinusoïdal à la fréquence d'une note de musique.

    Paramètres
    ----------
    note        : nom de la note (DO, RE, MI, FA, SOL, LA, SI — avec # pour dièse)
    octave      : octave (4 = octave centrale, LA4 = 440 Hz)
    duration    : durée totale en secondes
    sr          : taux d'échantillonnage
    amplitude   : amplitude crête (None → gain anti-saturation par défaut)
    save        : si True, sauvegarde dans sound/generated/
    """

    # Fréquences de référence à l'octave 4 (tempérament égal, LA4 = 440 Hz)
    _NOTES: dict[str, float] = {
        "DO":   261.63,
        "DO#":  277.18,
        "RE":   293.66,
        "RE#":  311.13,
        "MI":   329.63,
        "FA":   349.23,
        "FA#":  369.99,
        "SOL":  392.00,
        "SOL#": 415.30,
        "LA":   440.00,
        "LA#":  466.16,
        "SI":   493.88,
    }

    def __init__(
        self,
        note: str = "LA",
        octave: int = 4,
        duration: float = 10.0,
        sr: int = 44_100,
        amplitude: float | None = None,
        save: bool = False,
    ):
        self.note_name = note.upper().replace("B", "#")  # accepte DO#  ou DOb
        if self.note_name not in self._NOTES:
            raise ValueError(
                f"Note inconnue : '{note}'. "
                f"Notes disponibles : {list(self._NOTES.keys())}"
            )
        self.octave = octave
        self.duration = duration
        self.sr = sr
        self.amplitude = amplitude if amplitude is not None else _GAIN
        self.save = save
        self._signal: np.ndarray | None = None

        # LA4 = 440 Hz ; chaque octave double la fréquence
        self.frequency: float = self._NOTES[self.note_name] * (2 ** (octave - 4))

    def generate(self) -> np.ndarray:
        n_samples = int(self.sr * self.duration)
        t = np.linspace(0, self.duration, n_samples, endpoint=False)
        signal = (self.amplitude * np.sin(2 * np.pi * self.frequency * t)).astype(
            np.float32
        )
        self._signal = signal

        if self.save:
            self._write(
                f"note_{self.note_name}{self.octave}_{self.frequency:.2f}Hz_{self.duration}s.wav"
            )

        return signal

    def _write(self, filename: str) -> None:
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        path = SAVE_DIR / filename
        sf.write(str(path), self._signal, self.sr)
        print(f"[Note] Sauvegardé : {path}")


class BurstNote:
    """Rafales périodiques d'une note sinusoïdale (comme impulse_LA du notebook).

    Paramètres
    ----------
    note            : nom de la note musicale
    octave          : octave de la note
    burst_duration  : durée de chaque rafale en secondes
    repeat_every    : période de répétition en secondes
    duration        : durée totale du signal en secondes
    sr              : taux d'échantillonnage
    amplitude       : amplitude crête (None → gain anti-saturation)
    save            : si True, sauvegarde dans sound/generated/
    """

    def __init__(
        self,
        note: str = "LA",
        octave: int = 4,
        burst_duration: float = 1.0,
        repeat_every: float = 2.0,
        duration: float = 10.0,
        sr: int = 44_100,
        amplitude: float | None = None,
        save: bool = False,
    ):
        if burst_duration > repeat_every:
            raise ValueError("burst_duration doit être ≤ repeat_every")
        self._note = Note(note, octave, burst_duration, sr, amplitude, save=False)
        self.burst_duration = burst_duration
        self.repeat_every = repeat_every
        self.duration = duration
        self.sr = sr
        self.amplitude = self._note.amplitude
        self.save = save
        self._signal: np.ndarray | None = None

    @property
    def frequency(self) -> float:
        return self._note.frequency

    @property
    def note_name(self) -> str:
        return self._note.note_name

    @property
    def octave(self) -> int:
        return self._note.octave

    def generate(self) -> np.ndarray:
        n_total = int(self.sr * self.duration)
        signal = np.zeros(n_total, dtype=np.float32)

        burst = self._note.generate()
        period_samples = int(self.sr * self.repeat_every)

        for i in range(n_total // period_samples):
            start = i * period_samples
            end = start + len(burst)
            if end <= n_total:
                signal[start:end] = burst

        self._signal = signal

        if self.save:
            name = (
                f"burst_{self.note_name}{self.octave}"
                f"_{self.burst_duration}s_every{self.repeat_every}s"
                f"_{self.duration}s.wav"
            )
            self._write(name)

        return signal

    def _write(self, filename: str) -> None:
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        path = SAVE_DIR / filename
        sf.write(str(path), self._signal, self.sr)
        print(f"[BurstNote] Sauvegardé : {path}")
