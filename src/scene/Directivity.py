"""
Directivity.py
---------------
Directivité d'émission d'une source sonore : combien d'énergie/quel timbre
la source rayonne dans une direction donnée, relativement à son axe.

À ne pas confondre avec la HRTF (directivité de RÉCEPTION : comment la tête
de l'auditeur filtre selon la direction d'où arrive le son). Les deux sont
des filtres indépendants, appliqués en cascade sur des angles différents :

    x(t) --[Directivity: angle emission theta]--> --[HRTF: angle (az,el) recu]--> stereo

theta (émission) se calcule avec geometry.emission_angle_deg(), à partir de
l'axe d'émission de la source (cf. SourceOrientation.py) et du vecteur
source -> auditeur. (az, el) (réception) se calcule séparément, dans le
repère de la tête de l'auditeur (cf. Listener.py).

Classes
-------
Directivity                 (ABC) — interface commune
OmnidirectionalDirectivity  — aucune directivité (D=1 partout, référence)
CardioidDirectivity         — lobe cosinus^n, indépendant de la fréquence
PistonDirectivity           — piston circulaire encastré (Rayleigh),
                               dépendant de la fréquence — modèle physique
                               de référence pour haut-parleurs / bouche
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy.signal import fftconvolve
from scipy.special import j1


# ══════════════════════════════════════════════════════════════════════════════
# Classe de base abstraite
# ══════════════════════════════════════════════════════════════════════════════

class Directivity(ABC):
    """
    Interface commune à tous les modèles de directivité d'émission.

    magnitude(theta_deg, freqs_hz) est la seule méthode à implémenter :
    gain d'amplitude (linéaire, >= 0), normalisé à 1.0 sur l'axe (theta=0),
    pour chaque fréquence demandée. build_filter()/apply()/gain_db() sont
    dérivées automatiquement (template method).
    """

    @abstractmethod
    def magnitude(self, theta_deg: float, freqs_hz: np.ndarray) -> np.ndarray:
        """Gain d'amplitude linéaire D(theta, f), shape identique à freqs_hz."""
        ...

    def gain_db(self, theta_deg: float, freqs_hz: np.ndarray) -> np.ndarray:
        """Gain en dB : 20*log10(D(theta, f))."""
        return 20.0 * np.log10(np.maximum(self.magnitude(theta_deg, freqs_hz), 1e-10))

    def build_filter(self, theta_deg: float, sr: int, n_taps: int = 129) -> np.ndarray:
        """
        Construit un filtre FIR à phase linéaire approximant D(theta, f) à sr donné.

        Même technique que DistanceModel.compute_absorption_filter : calcul de
        la courbe de magnitude dans le domaine fréquentiel (rfft), IFFT,
        recentrage et fenêtrage de Hann. Pas de renormalisation d'énergie :
        la magnitude à f=0 vaut déjà 1.0 par construction (aucune directivité
        possible à fréquence nulle, quel que soit theta), donc la différence
        de niveau global entre deux angles est physique et doit être préservée.
        """
        n_fft = 2 ** int(np.ceil(np.log2(max(n_taps, 8))) + 1)
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
        H_mag = self.magnitude(theta_deg, freqs)

        h_full = np.fft.irfft(H_mag, n=n_fft)
        h_full = np.roll(h_full, n_taps // 2)[:n_taps]
        h_full *= np.hanning(n_taps)
        return h_full.astype(np.float32)

    def apply(
        self, signal: np.ndarray, theta_deg: float, sr: int, n_taps: int = 129
    ) -> np.ndarray:
        """
        Filtre un signal mono par la directivité à l'angle theta_deg.

        mode='same' : préserve la longueur du signal d'entrée (utile quand ce
        filtre doit ensuite être convolué avec une HRIR de longueur fixe,
        sans en changer la taille).
        """
        h = self.build_filter(theta_deg, sr, n_taps)
        sig = np.asarray(signal, dtype=np.float32)
        return fftconvolve(sig, h, mode="same").astype(np.float32)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


# ══════════════════════════════════════════════════════════════════════════════
# OmnidirectionalDirectivity — référence neutre
# ══════════════════════════════════════════════════════════════════════════════

class OmnidirectionalDirectivity(Directivity):
    """Source omnidirectionnelle : D(theta, f) = 1 partout (aucun effet)."""

    def magnitude(self, theta_deg: float, freqs_hz: np.ndarray) -> np.ndarray:
        return np.ones_like(np.asarray(freqs_hz, dtype=float))


# ══════════════════════════════════════════════════════════════════════════════
# CardioidDirectivity — lobe cosinus^n, indépendant de la fréquence
# ══════════════════════════════════════════════════════════════════════════════

class CardioidDirectivity(Directivity):
    """
    Directivité cardioïde simple : D(theta) = ((1 + cos(theta)) / 2) ** order.

    Indépendante de la fréquence (donc physiquement incomplète  une vraie
    source ne perd pas également toutes les fréquences hors-axe, cf.
    PistonDirectivity) mais utile comme modèle rapide/paramétrique.

    Paramètres
    ----------
    order : float
        Étroitesse du lobe. 1.0 = cardioïde classique. Plus grand = lobe
        plus étroit (source plus directive).
    floor_db : float
        Plancher de gain en dB, pour éviter un silence total à 180°
        (défaut : -40 dB).
    """

    def __init__(self, order: float = 1.0, floor_db: float = -40.0) -> None:
        self.order = float(order)
        self.floor_db = float(floor_db)

    def magnitude(self, theta_deg: float, freqs_hz: np.ndarray) -> np.ndarray:
        freqs_hz = np.asarray(freqs_hz, dtype=float)
        g = ((1.0 + np.cos(np.deg2rad(theta_deg))) / 2.0) ** self.order
        floor = 10.0 ** (self.floor_db / 20.0)
        return np.full_like(freqs_hz, max(g, floor))

    def __repr__(self) -> str:
        return f"CardioidDirectivity(order={self.order})"


# ══════════════════════════════════════════════════════════════════════════════
# PistonDirectivity — piston circulaire encastré (Rayleigh)
# ══════════════════════════════════════════════════════════════════════════════

class PistonDirectivity(Directivity):
    """
    Directivité d'un piston circulaire rigide encastré dans un baffle plan
    infini, en champ lointain (modèle de Rayleigh) :

        D(theta, f) = | 2 * J1(k*a*sin(theta)) / (k*a*sin(theta)) |

    avec k = 2*pi*f/c (nombre d'onde), a = rayon du piston, theta = angle
    par rapport à l'axe normal du piston (0° = sur l'axe).

    Comportement :
      - D(theta, 0 Hz) = 1 pour tout theta (aucune directivité en continu).
      - Basse fréquence (k*a << 1) : quasi omnidirectionnel.
      - Haute fréquence (k*a >> 1) : lobe étroit + zéros/lobes secondaires
        aux angles où k*a*sin(theta) = 3.8317, 7.0156, 10.1735, ... (zéros
        de J1  analogue acoustique de la tache d'Airy en optique).

    Paramètres
    ----------
    radius_m : float
        Rayon caractéristique de la source (m).
        Haut-parleur : rayon du diaphragme (souvent 0.03-0.15 m).
        Bouche/voix  : approximation usuelle 0.015-0.025 m.
    speed_of_sound : float
        Vitesse du son (m/s), défaut 343.0 (air, 20°C).
    floor_db : float
        Plancher de gain en dB (défaut -26 dB). Un piston rigide idéal a des
        zéros parfaits (annulation totale) ; une vraie source (peau, tissus,
        modes de rupture) n'a jamais des creux aussi profonds. Le plancher
        évite des "notches" artificiels et trop nets quand theta varie dans
        le temps (source qui tourne).
    """

    def __init__(
        self,
        radius_m: float,
        speed_of_sound: float = 343.0,
        floor_db: float = -26.0,
    ) -> None:
        if radius_m <= 0.0:
            raise ValueError(f"radius_m doit être > 0 (reçu : {radius_m}).")
        self.radius_m = float(radius_m)
        self.speed_of_sound = float(speed_of_sound)
        self.floor_db = float(floor_db)

    @staticmethod
    def _jinc(x: np.ndarray) -> np.ndarray:
        """2*J1(x)/x, avec la limite correcte (=1) en x=0 (sans 0/0)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            out = 2.0 * j1(x) / x
        return np.where(np.abs(x) < 1e-9, 1.0, out)

    def magnitude(self, theta_deg: float, freqs_hz: np.ndarray) -> np.ndarray:
        freqs_hz = np.asarray(freqs_hz, dtype=float)
        k = 2.0 * np.pi * freqs_hz / self.speed_of_sound
        x = k * self.radius_m * np.sin(np.deg2rad(theta_deg))
        D = np.abs(self._jinc(x))
        floor = 10.0 ** (self.floor_db / 20.0)
        return np.maximum(D, floor)

    def null_angle_deg(self, freq_hz: float, order: int = 1) -> float | None:
        """
        Angle du n-ième zéro de directivité à une fréquence donnée (degrés).

        Retourne None si k*a < premier zéro de J1 (pas de zéro atteignable
        à cette fréquence, quel que soit theta — source trop peu directive).
        """
        j1_zeros = [3.8317, 7.0156, 10.1735, 13.3237]  # zéros successifs de J1
        if order < 1 or order > len(j1_zeros):
            raise ValueError(f"order doit être entre 1 et {len(j1_zeros)}.")
        k = 2.0 * np.pi * freq_hz / self.speed_of_sound
        ka = k * self.radius_m
        ratio = j1_zeros[order - 1] / ka
        if ratio > 1.0:
            return None
        return float(np.degrees(np.arcsin(ratio)))

    def __repr__(self) -> str:
        return f"PistonDirectivity(radius_m={self.radius_m}, floor_db={self.floor_db})"
