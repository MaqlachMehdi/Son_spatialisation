# Spatialisation Sonore Binaurale — HRTF

Système de spatialisation audio 3D par convolution avec des HRTFs mesurées sur sujet humain. Prend un signal mono, une position dans l'espace (azimut, élévation, distance) et produit un fichier stéréo binaural — à écouter au casque.

---

## La base de données : IRCAM LISTEN

Le projet repose sur la base de données **IRCAM LISTEN**, développée à l'Institut de Recherche et Coordination Acoustique/Musique (Paris).

### Ce que contient ce dataset

Le dossier `dataset/` contient des mesures de **Head-Related Transfer Functions (HRTFs)** de 6 sujets humains (version compacte, 44 100 Hz) ainsi qu'une HRTF générique calculée par moyenne :

| Fichier | Description |
|---|---|
| `IRC_1002_C_44100.sofa` … `IRC_1057_C_44100.sofa` | 6 sujets IRCAM LISTEN individuels |
| `generic.sofa` | HRTF générique — moyenne pondérée des sujets (calculée par `HRTFGen`) ( téléchargé les autres pour une meilleur approximation) |

Une HRTF est la réponse en fréquence du trajet acoustique entre une source sonore et le tympan, pour une position donnée. Elle encode la façon dont la tête, les oreilles et le torse colorent le son selon la direction d'arrivée. En convolant un signal mono avec la paire gauche/droite correspondante à une position (az, el), on recrée la perception que la source vient de cette direction dans un casque.

### Structure du fichier SOFA

Le format SOFA (Spatially Oriented Format for Acoustics, AES69-2022) stocke :

| Variable | Forme | Description |
|---|---|---|
| `Data.IR` | `(M, 2, N)` | M paires de HRIRs × 2 oreilles × N échantillons |
| `SourcePosition` | `(M, 3)` | Position de chaque mesure `[azimut°, élévation°, distance_m]` |
| `Data.Delay` | `(M, 2)` | Retard fractionnaire par oreille |
| `Data.SamplingRate` | scalaire | 44 100 Hz |

### Grille de mesure IRC_1002

- **187 positions** de mesure réparties sur la sphère
- **Plan horizontal** (el=0°) : 72 positions tous les 5°
- **Élévations disponibles** : −45°, −30°, −15°, 0°, 15°, 30°, 45°, 60°, 75°, 90°
- **Distance de mesure** : r₀ = 2.06 m (microphone dans le conduit auditif)
- **Longueur des HRIRs** : 512 échantillons (~11.6 ms)

### Convention d'azimut

Le dataset IRCAM utilise une convention **horaire** (CW : 90° = droite), différente du standard SOFA (CCW : 90° = gauche). Le chargeur détecte automatiquement cette convention et convertit les azimuts au standard SOFA à l'initialisation :

```
[HRTF] Convention CW (90deg=droite) detectee -> azimuts convertis en CCW SOFA standard
```

---

## Architecture
```
src/
├── hrtf/
│   ├── hrtf.py               ← Chargement et accès au dataset SOFA
│   ├── hrtf_utils.py         ← Fonctions pures partagées (phase minimum, onset, reconstruction)
│   ├── HRTFInterpolator.py   ← Interpolation barycentrique sphérique (positions hors grille)
│   └── HRTFGen.py            ← HRTF générique par moyenne de N sujets SOFA
├── engine/
│   ├── Convolution.py        ← Convolution statique (source fixe)
│   ├── SegmentEngine.py      ← Convolution par blocs avec crossfade (windowing entrée)
│   ├── WOLAEngine.py         ← Convolution par blocs WOLA — reconstruction parfaite (Hann COLA)
│   └── DynamicConvolver.py   ← Convolution dynamique (SegmentEngine ou WOLAEngine)
├── scene/
│   ├── Soundsource.py        ← Dataclass source sonore positionnée
│   ├── DistanceModel.py      ← Modèle de propagation en champ libre
│   ├── Soundscape.py         ← Mix multi-source statique
│   ├── DynamicSoundscape.py  ← Mix multi-source dynamique
│   └── Trajectory.py         ← Trajectoires spatiales (5 types)
├── synthesis/
│   └── GenerateSound.py      ← Générateurs de signaux de test
└── analysis/
    ├── SoundVisu.py          ← Visualisations HRTF / ILD / ITD
    ├── SpatialisationVerif.py← Vérification objective de la spatialisation
    └── visualisation.py      ← Outils de visualisation complémentaires
```
---

## Méthodes implémentées

### 1. Chargement SOFA — `HRTF.from_sofa()`

Lecture du fichier SOFA via `netCDF4`. Détecte automatiquement :
- **L'ordre des oreilles** depuis `ReceiverPosition` (y > 0 = gauche)
- **La convention d'azimut** (CW vs CCW) par comparaison d'énergie à 90°

Sélection du voisin le plus proche via **distance grand-cercle** (correcte aux pôles, contrairement à la distance euclidienne sur les angles).

### 2. Interpolation HRTF — `HRTFInterpolator`

Pour les positions hors grille de mesure. Évite l'artefact de filtre en peigne de l'interpolation complexe directe.

**Pipeline :**
1. **Triangulation de Delaunay sphérique** via convex hull 3D (les positions HRTF forment une sphère convexe)
2. **Coordonnées barycentriques** (formule de Van Oosterom) — pondération angulaire correcte
3. **Interpolation magnitude + onset séparés** :
   - Magnitude : `M_interp(f) = Σ ωᵢ · |Hᵢ(f)|` — pas d'annulation spectrale
   - Retard d'onset : `τ_interp = Σ ωᵢ · τᵢ` — délai fractionnaire interpolé
   - Phase minimum reconstruite par cepstre réel (Oppenheim & Schafer) via `hrtf_utils`
   - Retard exact appliqué en fréquentiel : `H_final = H_mp · exp(−j2πf·τ_interp)`

**Drop-in replacement de `HRTF`** — même interface `get_hrir()`, utilisable partout sans modification.

### 3. Fonctions utilitaires partagées — `hrtf_utils`

Module de fonctions pures utilisé par `HRTFInterpolator` et `HRTFGen` :

| Fonction | Description |
|---|---|
| `build_mp_window(N)` | Fenêtre cepstrale pour la reconstruction phase-minimum |
| `detect_onset(hrir)` | Retard d'onset τ = argmax \|h(n)\| |
| `minimum_phase_from_magnitude(mag, N, w)` | Spectre complexe phase-min depuis un module |
| `reconstruct_hrir(mag, τ, N, w, freqs)` | Pipeline complet magnitude + onset → HRIR |

### 4. HRTF générique — `HRTFGen`

Construit une HRTF représentative en moyennant N sujets SOFA. Résout le problème de la HRTF non-individualisée sans nécessiter de nouvelles mesures.

**Algorithme (par position et par oreille) :**
```
M_avg(f) = Σᵢ wᵢ · |RFFT(hᵢ(f))|   — moyenne des modules (jamais d'annulation spectrale)
τ_avg    = Σᵢ wᵢ · τᵢ               — onset moyen pondéré
H_mp     = MinimumPhase(M_avg)        — reconstruction via cepstre réel
H_final  = H_mp · exp(−j2πf·τ_avg)  — retard fractionnaire exact
```

Contrairement à la moyenne de spectres complexes (risque de filtre en peigne dû aux différences d'ITD inter-sujets), la moyenne des modules garantit M_avg > 0 — aucune annulation spectrale possible.

**Usage :**
```python
# Calcul une seule fois (~30 s pour 6 sujets)
gen = HRTFGen.from_directory("dataset/", pattern="IRC_*.sofa")
gen.save("dataset/generic.sofa")

# Utilisation rapide (rechargement instantané depuis le cache SOFA)
gen    = HRTFGen.from_sofa("dataset/generic.sofa")
interp = HRTFInterpolator(gen)   # interpolation sur la HRTF générique
conv   = DynamicConvolver(hrtf=interp, ..., hop_ms=25.0)
```

**Drop-in replacement de `HRTF`** — même interface, compatible avec `HRTFInterpolator` et `DynamicConvolver`.

### 5. Convolution statique — `HRTFConvolver`

Convolution fréquentielle via `fftconvolve` (scipy). Accepte un signal depuis fichier WAV ou depuis un array numpy.

```
signal mono × HRIR_gauche → canal gauche
signal mono × HRIR_droite → canal droit
```

Normalisation binaural conjointe (même facteur pour L et R, préserve l'équilibre interaural).

### 6. Modèle de distance — `DistanceModel`

Étend les HRTFs mesurées à r₀ = 2.06 m vers une distance cible r quelconque. Trois effets physiques :

| Effet | Formule | Activation |
|---|---|---|
| Atténuation 1/r | `gain = r₀ / r` | Toujours |
| Retard de propagation | `Δt = (r − r₀) / c` | Si `r > r₀` |
| Absorption atmosphérique | ISO 9613-1 (FIR) | Optionnel, pertinent pour r > 50 m |

Note : le son dans l'air est non-dispersif (pas d'étalement temporel fréquence-dépendant en champ libre).

### 7. Paysage sonore statique — `Soundscape`

Combine N sources statiques spatialisées indépendamment puis mixées :
- Convolution parallèle de chaque source
- Zero-padding pour aligner les longueurs
- Somme et normalisation conjointe

### 8. Trajectoires — `Trajectory`

Convertit un instant `t` (secondes) en position angulaire `(az°, el°)`. Cinq types :

| Classe | Description |
|---|---|
| `CircularTrajectory` | Rotation à vitesse constante, élévation fixe |
| `EllipseTrajectory` | Trajectoire elliptique avec demi-axes azimut et élévation configurables |
| `LinearTrajectory` | Interpolation sphérique entre deux positions (SLERP) |
| `RectilinearTrajectory` | Déplacement en ligne droite dans l'espace 3D (distance variable) |
| `CustomTrajectory` | Points de passage libres avec interpolation cubique |

Toutes les trajectoires exposent `plot()` (vue temporelle + vue polaire avec gradient de temps).

### 9. Convolution dynamique — `DynamicConvolver` + moteurs

Pour les sources en mouvement. `DynamicConvolver` orchestre la trajectoire et délègue la convolution à l'un des deux moteurs :

#### SegmentEngine — Overlap-Add avec fenêtrage d'entrée

```python
conv = DynamicConvolver(hrtf=hrtf, signal=signal, sr=sr, trajectory=traj,
                         segment_ms=50.0, overlap_ms=15.0, crossfade_type="cosine")
```

- Découpe le signal en blocs de `segment_ms` avec look-ahead de `overlap_ms`
- **Fenêtrage sur l'entrée** (pas sur la sortie) : les mêmes échantillons reçoivent des poids complémentaires dans deux segments adjacents → équivalent à une interpolation implicite des HRIRs, sans filtre en peigne
- Trois enveloppes disponibles : `linear`, `cosine`, `equal_power`

#### WOLAEngine — Weighted Overlap-Add (COLA Hann)

```python
conv = DynamicConvolver(hrtf=hrtf, signal=signal, sr=sr, trajectory=traj,
                         hop_ms=25.0)
```

- Paramètre unique `hop_ms` (pas de choix d'enveloppe)
- Fenêtre de Hann périodique (`sym=False`) : `w[n] + w[n + hop] = 1.0` exactement (**propriété COLA**)
- Reconstruction parfaite garantie si la HRTF est stationnaire
- Recommandé pour les rotations rapides ou les trajectoires avec fort changement d'ITD

**Règle pour `hop_ms`** : `hop_ms = WOLAEngine.optimal_hop_ms(period_s, angular_resolution_deg=5)`.

### 10. Paysage sonore dynamique — `DynamicSoundscape`

N sources avec chacune leur propre trajectoire. Rendu indépendant puis mix normalisé.

### 11. Vérification — `SpatialisationVerif`

Vérifie objectivement la cohérence de la spatialisation :
- **ILD** (Interaural Level Difference) en dB sur tous les azimuts
- **ITD** (Interaural Time Delay) en millisecondes sur tous les azimuts
- Comparaison avec le modèle théorique de Woodworth (tête sphérique)

---

## Exemples d'utilisation

### Source fixe
```python
# Source fixe
from hrtf import HRTF
from engine import HRTFConvolver

hrtf = HRTF.from_sofa("dataset/IRC_1002_C_44100.sofa")
conv = HRTFConvolver(hrtf, azimuth=45.0, elevation=0.0)
output = conv.convolve_file("signal.wav")

```

### Source en mouvement (WOLAEngine)
```python
# Source en mouvement (WOLAEngine)
from hrtf import HRTF, HRTFInterpolator
from scene import CircularTrajectory
from engine import DynamicConvolver
import soundfile as sf

hrtf   = HRTF.from_sofa("dataset/generic.sofa")
interp = HRTFInterpolator(hrtf)
traj   = CircularTrajectory(duration_s=10.0, period_s=4.0, elevation=22.0)
conv   = DynamicConvolver(hrtf=interp, signal=signal, sr=44100,
                           trajectory=traj, hop_ms=25.0)
output = conv.run()   # shape (N, 2)
sf.write("output.wav", output, 44100)
```

### HRTF générique depuis plusieurs sujets
```python
# HRTF générique depuis plusieurs sujets
from hrtf import HRTFGen

gen = HRTFGen.from_directory("dataset/", pattern="IRC_*.sofa")
gen.save("dataset/generic.sofa")
gen.plot_comparison(other=hrtf_individual, azimuth=90.0, elevation=0.0)
```

---

## Convention angulaire

Tout le code utilise la **convention SOFA** :

| Direction | Azimut | Élévation |
|---|---|---|
| Devant | 0° | 0° |
| Gauche | 90° | 0° |
| Derrière | 180° | 0° |
| Droite | 270° | 0° |
| Dessus | — | 90° |
| Dessous | — | −90° |

---

## Structure des fichiers

```
son_spatialisation/
├── dataset/
│   ├── IRC_1002_C_44100.sofa
│   ├── IRC_1003_C_44100.sofa
│   ├── IRC_1015_C_44100.sofa
│   ├── IRC_1042_C_44100.sofa
│   ├── IRC_1048_C_44100.sofa
│   ├── IRC_1057_C_44100.sofa
│   └── generic.sofa
├── src/
│   ├── hrtf/
│   │   ├── hrtf.py
│   │   ├── hrtf_utils.py
│   │   ├── HRTFInterpolator.py
│   │   └── HRTFGen.py
│   ├── engine/
│   │   ├── Convolution.py
│   │   ├── SegmentEngine.py
│   │   ├── WOLAEngine.py
│   │   └── DynamicConvolver.py
│   ├── scene/
│   │   ├── Soundsource.py
│   │   ├── DistanceModel.py
│   │   ├── Soundscape.py
│   │   ├── DynamicSoundscape.py
│   │   └── Trajectory.py
│   ├── synthesis/
│   │   └── GenerateSound.py
│   └── analysis/
│       ├── SoundVisu.py
│       ├── SpatialisationVerif.py
│       └── visualisation.py
├── sound/generated/
├── notebooks/
│   ├── static HRTF Demo.ipynb
│   ├── DynamicConvolver_viz.ipynb
│   ├── DynamicSoundscape_viz.ipynb
│   ├── HRTFGen.ipynb
│   ├── SegmentEngine_Viz.ipynb
│   └── traj_viz.ipynb
├── docs/
│   └── hrtf_interpolation.tex
├── pyproject.toml
└── Requirements.txt
```

---

## Dépendances

| Package | Usage |
|---|---|
| `numpy` | Calcul numérique |
| `scipy` | FFT, convolution, géométrie (ConvexHull), fenêtres WOLA |
| `soundfile` | Lecture/écriture WAV |
| `netCDF4` | Lecture/écriture fichiers SOFA |
| `matplotlib` | Visualisations |
| `librosa` | Rééchantillonnage |
| `jupyter` | Notebooks interactifs |

```bash
pip install -r Requirements.txt
```

---

## Référence

Base de données IRCAM LISTEN :
> UMR 9912 - STMS - IRCAM/CNRS/UPMC. *LISTEN HRTF Database* — Olivier Warusfel.
> http://recherche.ircam.fr/equipes/salles/listen/

Licence IRCAM : utilisation libre à des fins éducatives, de recherche ou commerciales. Toute reproduction doit inclure la notice de copyright IRCAM.
