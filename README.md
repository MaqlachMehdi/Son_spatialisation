# Spatialisation Sonore Binaurale — HRTF

Système de spatialisation audio 3D par convolution avec des HRTFs mesurées sur sujet humain. Prend un signal mono, une position dans l'espace (azimut, élévation, distance) et produit un fichier stéréo binaural — à écouter au casque.

---

## Écouter les exemples *(casque recommandé)*

<table>
<tr>
<td align="center"><b>Cercle hors grille</b><br><sub>HRTFInterpolator — rotation circulaire<br>sur positions hors grille de mesure</sub><br><audio controls><source src="sound/generated/cercle_hors_grille.wav" type="audio/wav"></audio></td>
<td align="center"><b>Ping-pong Do→La</b><br><sub>DynamicConvolver WOLAEngine —<br>source en mouvement gauche/droite</sub><br><audio controls><source src="sound/generated/pingpong_do_la.wav" type="audio/wav"></audio></td>
<td align="center"><b>Tchaïkovski — scène complète</b><br><sub>InstrumentSpatializer — orchestre entier<br>reconstruit depuis The Spheres dataset</sub><br><audio controls><source src="sound/tchaikovsky_full.wav" type="audio/wav"></audio></td>
</tr>
</table>

---

## Nouveautés — Reconstruction orchestrale The Spheres

### Structure du dataset The Spheres

Le dataset **The Spheres** (enregistrements orchestraux multi-micros) est organisé ainsi :

```
dataset_live/Tchaikovsky/
├── Violin_1/          ← MICROPHONE placé près des violons
│   ├── Violin_1_1.flac   → violon 1 capté par ce micro  (proche, fort)
│   ├── Trumpet_1.flac    → trompette captée par ce micro (lointain, faible)
│   └── ...
├── Trumpet/           ← MICROPHONE placé près des trompettes
│   ├── Violin_1_1.flac   → violon 1 capté par ce micro  (lointain, faible)
│   ├── Trumpet_1.flac    → trompette captée par ce micro (proche, fort)
│   └── ...
```

> **Règle clé** : le **dossier** = position du microphone, le **fichier** = source instrumentale captée.  
> Chaque instrument a été enregistré en isolation. Chaque fichier = une paire source/micro.

### Module `src/instrument/` — InstrumentSpatializer

Pipeline de reconstruction binaurale par stem :

```
Pour chaque dossier-micro qui contient le fichier du stem :
  1. Charge l'audio (source isolée captée depuis ce micro)
  2. Récupère la position 3D du dossier-micro (via mic_folder_map)
  3. Convolue avec la paire HRIR correspondante (HRTFInterpolator)
  4. Applique le DistanceModel (atténuation 1/r + retard)
  5. Accumule → mix binaural du stem
```

**Un WAV par stem — séparation préservée :**

```
render_instrument("Bassoon")  →  Bassoon_1_spatial.wav
                                 Bassoon_2_spatial.wav

render_instrument("Horn")     →  Horn_1_spatial.wav
                                 Horn_2_spatial.wav
                                 Horn_3_spatial.wav
                                 Horn_4_spatial.wav
```

**Usage :**
```python
from src.instrument import InstrumentSpatializer, RenderConfig

spatializer = InstrumentSpatializer(
    hrtf_path           = "dataset/generic.sofa",
    positions_json      = "dataset_live/positions_phase1.json",
    channel_map_json    = "dataset_live/channel_map_Tchaikovsky.json",
    mic_folder_map_json = "dataset_live/mic_folder_map_Tchaikovsky.json",
    dataset_root        = "dataset_live",
    piece               = "Tchaikovsky",
    config              = RenderConfig(normalize=False),
)

# Un stem isolé
mix = spatializer.render_stem("Violin_1_1", output_path="sound/violin_1_1.wav")

# Toute une famille d'instruments
spatializer.render_instrument("Horn", output_dir="sound/scene/")

# Scène complète
for instrument in spatializer.available_instruments():
    spatializer.render_instrument(instrument, output_dir="sound/tchaikovsky_scene/")
```

**Mix final :**
```python
import soundfile as sf
from src.instrument import InstrumentSpatializer

stem_files = sorted(Path("sound/tchaikovsky_scene").glob("*_spatial.wav"))
signals = [sf.read(str(f), dtype="float32", always_2d=True)[0] for f in stem_files]

mix  = InstrumentSpatializer.mix_stereo(signals)
peak = max(abs(mix.max()), abs(mix.min())) + 1e-10
sf.write("sound/tchaikovsky_full.wav", (mix / peak).astype("float32"), 48000)
```

**Exemple — scène Tchaïkovski complète :**
<audio controls><source src="sound/tchaikovsky_full.wav" type="audio/wav"></audio>

---

## La base de données : IRCAM LISTEN

Le projet repose sur la base de données **IRCAM LISTEN**, développée à l'Institut de Recherche et Coordination Acoustique/Musique (Paris).

### Ce que contient ce dataset

Le dossier `dataset/` contient des mesures de **Head-Related Transfer Functions (HRTFs)** de 6 sujets humains (version compacte, 44 100 Hz) ainsi qu'une HRTF générique calculée par moyenne :

| Fichier | Description |
|---|---|
| `IRC_1002_C_44100.sofa` … `IRC_1057_C_44100.sofa` | 6 sujets IRCAM LISTEN individuels |
| `generic.sofa` | HRTF générique — moyenne pondérée des sujets (calculée par `HRTFGen`) |

Une HRTF encode la façon dont la tête, les oreilles et le torse colorent le son selon la direction d'arrivée. En convolant un signal mono avec la paire gauche/droite correspondante à une position (az, el), on recrée la perception que la source vient de cette direction dans un casque.

### Structure du fichier SOFA

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
- **Distance de mesure** : r₀ = 2.06 m
- **Longueur des HRIRs** : 512 échantillons (~11.6 ms)

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
├── instrument/
│   ├── InstrumentSpatializer.py  ← Reconstruction binaurale par stem (The Spheres)
│   ├── recording_repo.py         ← Accès aux fichiers audio du dataset
│   ├── position_provider.py      ← Résolution stem/dossier → position 3D
│   └── models.py                 ← ListenerConfig, RenderConfig, MicRecording
├── multimodal/
│   ├── multimodal_dataset.py ← Dataset cross-modal oreilles ↔ HRTF (build / load)
│   ├── ear_processor.py      ← Chargement et prétraitement des photos d'oreilles
│   ├── hrtf_processor.py     ← Extraction des coefficients SH depuis SOFA
│   ├── subject_record.py     ← Dataclass par sujet (chemin SOFA + images)
│   ├── sh_transform.py       ← SphericalHarmonicsTransform (encode / decode)
│   ├── sh_visualizer.py      ← Visualisation des coefficients SH
│   └── augmentation/
│       ├── base.py               ← Interface AugmentationBase
│       ├── azimuth_reflection.py ← Réflexion azimutale (L↔R + miroir SH)
│       ├── image_transforms.py   ← Augmentations image (flip, bruit, contraste)
│       └── pipeline.py           ← AugmentationPipeline — compose et applique
├── models/
│   ├── ear_encoder.py        ← EarEncoder (EfficientNet-B0 siamois + tête MLP)
│   ├── hrtf_encoder.py       ← HRTFEncoder (SH → vecteur latent)
│   ├── contrastive_model.py  ← ContrastiveModel (cross-modal NT-Xent)
│   ├── nt_xent.py            ← Loss NT-Xent standard
│   ├── supervised_nt_xent.py ← Loss NT-Xent supervisée (positifs explicites)
│   └── trainer.py            ← Trainer 3 phases — TensorBoard + MLflow
├── inference/
│   ├── gallery.py            ← Gallery (base HRTF de référence, k-NN cosinus)
│   └── predictor.py          ← HRTFPredictor (photo → HRTF personnalisée)
├── evaluation/
│   ├── recall_callback.py    ← RecallCallback (Recall@1 à chaque epoch)
│   ├── validation_report.py  ← ValidationReport (Recall@k, MRR, LSD)
│   ├── lsd.py                ← Log-Spectral Distortion
│   ├── metrics.py            ← recall_at_k, mrr
│   ├── embedding_space.py    ← Visualisation de l'espace latent
│   └── visualizer.py         ← Courbes d'entraînement et matrices de similarité
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

Sélection du voisin le plus proche via **distance grand-cercle**.

### 2. Interpolation HRTF — `HRTFInterpolator`

Pour les positions hors grille de mesure. Évite l'artefact de filtre en peigne de l'interpolation complexe directe.

**Pipeline :**
1. **Triangulation de Delaunay sphérique** via convex hull 3D
2. **Coordonnées barycentriques** (formule de Van Oosterom)
3. **Interpolation magnitude + onset séparés** :
   - Magnitude : `M_interp(f) = Σ ωᵢ · |Hᵢ(f)|`
   - Retard d'onset : `τ_interp = Σ ωᵢ · τᵢ`
   - Phase minimum reconstruite par cepstre réel

**Exemple — rotation circulaire hors grille :**
<audio controls><source src="sound/generated/cercle_hors_grille.wav" type="audio/wav"></audio>

### 3. Fonctions utilitaires partagées — `hrtf_utils`

| Fonction | Description |
|---|---|
| `build_mp_window(N)` | Fenêtre cepstrale pour la reconstruction phase-minimum |
| `detect_onset(hrir)` | Retard d'onset τ = argmax \|h(n)\| |
| `minimum_phase_from_magnitude(mag, N, w)` | Spectre complexe phase-min depuis un module |
| `reconstruct_hrir(mag, τ, N, w, freqs)` | Pipeline complet magnitude + onset → HRIR |

### 4. HRTF générique — `HRTFGen`

Construit une HRTF représentative en moyennant N sujets SOFA.

```
M_avg(f) = Σᵢ wᵢ · |RFFT(hᵢ(f))|   — moyenne des modules
τ_avg    = Σᵢ wᵢ · τᵢ               — onset moyen pondéré
H_mp     = MinimumPhase(M_avg)        — reconstruction via cepstre réel
H_final  = H_mp · exp(−j2πf·τ_avg)  — retard fractionnaire exact
```

### 5. Convolution statique — `HRTFConvolver`

Convolution fréquentielle via `fftconvolve`. Normalisation binaural conjointe (même facteur L et R).

### 6. Modèle de distance — `DistanceModel`

| Effet | Formule | Activation |
|---|---|---|
| Atténuation 1/r | `gain = r₀ / r` | Toujours |
| Retard de propagation | `Δt = (r − r₀) / c` | Si `r > r₀` |
| Absorption atmosphérique | ISO 9613-1 (FIR) | Optionnel |

### 7. Paysage sonore statique — `Soundscape`

Combine N sources statiques spatialisées indépendamment puis mixées.

**Exemple — Do + Si spatialisés en positions fixes :**
<audio controls><source src="sound/generated/soundscape_do_si.wav" type="audio/wav"></audio>

### 8. Trajectoires — `Trajectory`

| Classe | Description |
|---|---|
| `CircularTrajectory` | Rotation à vitesse constante, élévation fixe |
| `EllipseTrajectory` | Trajectoire elliptique avec demi-axes configurables |
| `LinearTrajectory` | Interpolation sphérique SLERP entre deux positions |
| `RectilinearTrajectory` | Déplacement en ligne droite (distance variable) |
| `CustomTrajectory` | Points de passage libres avec interpolation cubique |

**Exemple — trajectoire elliptique :**
<audio controls><source src="sound/generated/ellipse_interp.wav" type="audio/wav"></audio>

### 9. Convolution dynamique — `DynamicConvolver` + moteurs

#### SegmentEngine — Overlap-Add avec fenêtrage d'entrée

```python
conv = DynamicConvolver(hrtf=hrtf, signal=signal, sr=sr, trajectory=traj,
                         segment_ms=50.0, overlap_ms=15.0, crossfade_type="cosine")
```

#### WOLAEngine — Weighted Overlap-Add (COLA Hann)

```python
conv = DynamicConvolver(hrtf=hrtf, signal=signal, sr=sr, trajectory=traj, hop_ms=25.0)
```

Fenêtre de Hann périodique : `w[n] + w[n + hop] = 1.0` exactement — reconstruction parfaite garantie.

**Exemple — ping-pong Do→La (source en mouvement) :**
<audio controls><source src="sound/generated/pingpong_do_la.wav" type="audio/wav"></audio>

### 10. Paysage sonore dynamique — `DynamicSoundscape`

N sources avec chacune leur propre trajectoire. Rendu indépendant puis mix normalisé.

### 11. Spatialisation orchestrale — `InstrumentSpatializer`

Reconstruction binaurale depuis le dataset The Spheres. Un WAV par stem, tous les dossiers-micros utilisés pour l'image spatiale.

Voir section [Nouveautés](#nouveautés----reconstruction-orchestrale-the-spheres) pour le détail.

### 12. Vérification — `SpatialisationVerif`

- **ILD** (Interaural Level Difference) en dB sur tous les azimuts
- **ITD** (Interaural Time Delay) en ms sur tous les azimuts
- Comparaison avec le modèle théorique de Woodworth

---

## Module cross-modal — personnalisation HRTF par photo d'oreille

L'objectif de ce module est d'apprendre à aligner deux modalités dans un espace latent partagé : une **photo d'oreille gauche + droite** d'un côté, et une **HRTF mesurée** de l'autre. Une fois entraîné, le modèle peut retrouver par similarité cosinus la HRTF la plus proche d'un nouvel utilisateur à partir de ses seules photos.

```
Photo oreille G  ──┐                    ┌── HRTF sujet (SH)
Photo oreille D  ──┤── EarEncoder ─────┤
                   │      z_ear [128]   └── HRTFEncoder ── z_hrtf [128]
                   │                              │
                   └── NT-Xent loss ─────────────┘
                       (aligne z_ear ↔ z_hrtf du même sujet)
```

### 13. Représentation — `SphericalHarmonicsTransform`

Les HRTFs sont projetées sur une base d'**harmoniques sphériques** (SH) d'ordre L. Cela compresse la dépendance angulaire en `(L+1)²` coefficients, réduit le bruit de mesure et permet de reconstruire la HRTF à n'importe quelle direction.

```python
sh = SphericalHarmonicsTransform(order=5)   # 36 coefficients SH

coeffs = sh.encode_subject("IRC_1002_C_44100.sofa")  # [36, 129, 2]
hrtf   = sh.decode(coeffs, positions_deg=my_grid)    # [N_pos, 129, 2]
```

| Paramètre | Valeur typique | Description |
|---|---|---|
| `order` | 5 ou 10 | Ordre SH — (L+1)² = 36 ou 121 coefficients |
| `n_freqs` | 129 | Bins FFT (RFFT de 256 échantillons) |
| Sorties | `[n_sh, n_freqs, 2]` | Coefficients pour oreille G et D |

### 14. EarEncoder — `EfficientNetB0` siamois

Backbone **EfficientNet-B0** pré-entraîné ImageNet, poids **partagés** entre oreille gauche et droite. Les deux features maps sont concaténées puis projetées vers un embedding L2-normalisé de dimension 128.

```
ear_left  [B, 224, 224, 3] ──┐
                               ├── EfficientNetB0 (shared) ── GAP ── concat [B, 2560]
ear_right [B, 224, 224, 3] ──┘                                          │
                                               Dense(512, relu) → Dropout(0.3)
                                               Dense(256, relu)
                                               Dense(128)
                                               L2-normalize
                                               z_ear [B, 128]
```

**Trois phases de fine-tuning :**

| Phase | Backbone | Params entraînables | LR |
|---|---|---|---|
| 1 | Gelé | ~1.4 M (head seulement) | 1e-3 |
| 2 | 20 dernières couches dégelées | ~5 M | 1e-4 |
| 3 | Entièrement dégelé | ~6.5 M | 1e-5 |

### 15. HRTFEncoder

Prend les coefficients SH `[n_sh, n_freqs, 2]` et les projette dans le même espace latent que `EarEncoder`.

```
[B, n_sh, n_freqs, 2]
    │  Reshape → [B, n_sh, n_freqs×2]
    │  Dense(32, relu)  — partagé sur les n_sh lignes
    │  Flatten → [B, n_sh × 32]
    │  Dense(256, relu) → Dropout → Dense(128) → L2-normalize
z_hrtf [B, 128]
```

### 16. ContrastiveModel — NT-Xent cross-modal

Le `ContrastiveModel` combine les deux encodeurs et minimise la **NT-Xent loss** cross-modale : dans un batch de N paires, la paire `(z_ear_i, z_hrtf_i)` est le seul positif pour les 2(N−1) négatifs restants.

Deux modes de loss disponibles :

| Mode | Loss | Quand l'utiliser |
|---|---|---|
| `supervised=False` | NT-Xent standard | Toutes les augmentations sont des négatifs |
| `supervised=True` | Supervised NT-Xent | Les augmentations du même sujet sont des positifs explicites |

```python
from src.models import ContrastiveModel, EarEncoder, HRTFEncoder

enc_ear  = EarEncoder(embedding_dim=128)
enc_hrtf = HRTFEncoder(n_sh=36, n_freqs=129, embedding_dim=128)
model    = ContrastiveModel(enc_ear, enc_hrtf, temperature=0.07, supervised=True)
model.compile(optimizer='adam')
```

### 17. MultimodalDataset + augmentation

`MultimodalDataset` charge l'ensemble des sujets depuis une base combinant fichiers SOFA + photos d'oreilles, applique le preprocessing SH et image, puis découpe en train/val/test.

L'**`AugmentationPipeline`** est appliquée **offline avant le split** pour éviter la fuite de données inter-augmentation, et à nouveau **online** via `tf.data.Dataset.map()` sur le train set uniquement.

Augmentation disponible : **`AzimuthalReflection`** permute oreille G↔D et applique le miroir de phase sur les coefficients SH azimutaux. Permet de doubler le dataset sans biais.

```python
from src.multimodal import MultimodalDataset
from src.multimodal.augmentation import AugmentationPipeline, AzimuthalReflection

pipeline = AugmentationPipeline([AzimuthalReflection(sh_order=5, p=0.5)])
ds = MultimodalDataset.build("dataset_crossmodal/", sh_order=5, augmentation=pipeline)

ds.train   # tf.data.Dataset — augmenté
ds.val     # tf.data.Dataset — non augmenté
ds.test    # tf.data.Dataset — non augmenté
```

---

## Pipeline d'entraînement — `Trainer`

`Trainer` orchestre les 3 phases de fine-tuning avec **TensorBoard** et **MLflow** connectés en parallèle. Il gère automatiquement les checkpoints, l'early stopping, et génère un rapport de validation à la fin de chaque phase.

```python
from src.models import Trainer

trainer = Trainer(
    model,
    enc_ear,
    experiment   = 'temp007_bs32',
    dataset_path = 'dataset/preprocessed_dataset.npz',
    base_dir     = 'checkpoints/',
    dataset      = ds,          # active RecallCallback si fourni
)

trainer.run_phase(1, ds.train, ds.val, epochs=50)
trainer.run_phase(2, ds.train, ds.val, epochs=30)
trainer.run_phase(3, ds.train, ds.val, epochs=20)
trainer.end()
trainer.plot_history()
```

**Callbacks actifs à chaque phase :**

| Callback | Rôle |
|---|---|
| `RecallCallback` | Calcule Recall@1 oreille→HRTF à chaque epoch |
| `MLflowEpochLogger` | Logue train/val loss + Recall@1 dans MLflow |
| `TensorBoard` | Histogrammes des poids + courbes en temps réel |
| `EarlyStopping` | Arrêt si val_loss ne descend plus (patience=10) |
| `ReduceLROnPlateau` | Divise le LR par 2 si val_loss stagne (patience=5) |
| `ModelCheckpoint` | Sauvegarde `.weights.h5` du meilleur epoch |

**Lancer les UIs de suivi :**
```bash
# TensorBoard
tensorboard --logdir checkpoints/exp_.../logs --port 6007

# MLflow
mlflow ui --backend-store-uri checkpoints/mlruns.db --port 5000
# → http://localhost:5000
```

**Artefacts sauvegardés par phase :**
- `phase1_best.weights.h5` meilleurs poids Keras
- `phase1_config.json` paramètres + métriques (loggé dans MLflow)

### RecallCallback — évaluation en temps réel

À chaque epoch, la gallery est reconstruite depuis le **train set uniquement** (embeddings HRTF), puis les photos du **val set** sont encodées et matchées par similarité cosinus. Le critère de succès est `base_id` identique (ex : `H10_mirror` → `H10` = correct) pour mitiger la fuite partielle liée à l'augmentation.

```
val_recall@1  loggé dans MLflow + ajouté aux logs Keras
```

---

## Pipeline d'inférence — `HRTFPredictor` + `Gallery`

### Gallery

La `Gallery` est une base de référence construite une seule fois depuis le train set : elle stocke les embeddings HRTF `z_hrtf` L2-normalisés et les coefficients SH bruts de chaque sujet connu. Elle peut être sauvegardée et rechargée sans recalculer les embeddings.

```python
from src.inference import Gallery

gallery = Gallery.build(model, dataset, split='train')
gallery.save('checkpoints/gallery.npz')

# Rechargement
gallery = Gallery.load('checkpoints/gallery.npz')
subject_id, hrtf_coeffs, score = gallery.find_nearest(z_ear)
```

La recherche du plus proche voisin est un simple produit scalaire `z_ear · z_hrtf.T` (cosinus sur vecteurs L2-normalisés).

### HRTFPredictor

`HRTFPredictor` donne deux photos d'oreilles, récupérer une HRTF personnalisée prête à l'emploi.

```python
from src.inference import HRTFPredictor

predictor = HRTFPredictor.build(model, dataset, sh_order=5)
predictor.save_gallery('checkpoints/gallery.npz')

# Depuis des chemins d'images
result = predictor.predict('ear_L.png', 'ear_R.png')
print(result['subject_id'])   # 'H10'
print(result['similarity'])   # 0.87  — score cosinus ∈ [-1, 1]
print(result['hrtf_coeffs'])  # [36, 129, 2]

# HRTF décodée aux positions voulues
hrtf = predictor.predict_at_positions(
    'ear_L.png', 'ear_R.png',
    positions_deg=my_grid,   # [N_pos, 2]  [azimut, élévation]
)
# hrtf : [N_pos, 129, 2]  → peut être injecté dans HRTFInterpolator
```

---

## Évaluation

### ValidationReport

Rapport complet calculé automatiquement par `Trainer` à la fin de chaque phase, ou déclenché manuellement.

| Métrique | Description |
|---|---|
| **Recall@1** | La HRTF récupérée appartient-elle au bon sujet ? |
| **Recall@3** | Le bon sujet est-il dans les 3 premières réponses ? |
| **MRR** | Mean Reciprocal Rank — rang moyen de la bonne paire |
| **S+ / S−** | Similarité cosinus moyenne paires correctes / incorrectes |
| **Ratio S+/S−** | Discrimination — valeur cible > 1.5 |
| **LSD (dB)** | Log-Spectral Distortion entre HRTF vraie et récupérée |

```python
from src.evaluation import ValidationReport

report = ValidationReport(model, dataset)
results = report.run(phase=2, log_mlflow=True)
# → logue toutes les métriques dans MLflow
```

---

## Exemples d'utilisation

### Source fixe
```python
from src.hrtf import HRTF
from src.engine import HRTFConvolver

hrtf   = HRTF.from_sofa("dataset/IRC_1002_C_44100.sofa")
conv   = HRTFConvolver(hrtf, azimuth=45.0, elevation=0.0)
output = conv.convolve_file("signal.wav")
```

### Source en mouvement (WOLAEngine)
```python
from src.hrtf import HRTF, HRTFInterpolator
from src.scene import CircularTrajectory
from src.engine import DynamicConvolver
import soundfile as sf

hrtf   = HRTF.from_sofa("dataset/generic.sofa")
interp = HRTFInterpolator(hrtf)
traj   = CircularTrajectory(duration_s=10.0, period_s=4.0, elevation=22.0)
conv   = DynamicConvolver(hrtf=interp, signal=signal, sr=44100,
                           trajectory=traj, hop_ms=25.0)
output = conv.run()
sf.write("output.wav", output, 44100)
```

### Violon spatialisé (InstrumentSpatializer)

**Violin_1 — reconstruction binaurale depuis enregistrement multi-micros :**
<audio controls><source src="sound/violin_1_spatial.wav" type="audio/wav"></audio>

### HRTF générique depuis plusieurs sujets
```python
from src.hrtf import HRTFGen

gen = HRTFGen.from_directory("dataset/", pattern="IRC_*.sofa")
gen.save("dataset/generic.sofa")
```

---

## Structure des fichiers

```
son_spatialisation/
├── dataset/                          ← HRTFs IRCAM LISTEN (non versionnées)
├── dataset_live/
│   ├── Tchaikovsky/                  ← The Spheres dataset (non versionné)
│   ├── channel_map_Tchaikovsky.json  ← stem → label position
│   ├── mic_folder_map_Tchaikovsky.json ← dossier-micro → label position
│   └── positions_phase1.json         ← positions 3D MDS des micros
├── src/
│   ├── hrtf/
│   ├── engine/
│   ├── scene/
│   ├── instrument/
│   ├── multimodal/                   ← dataset cross-modal + augmentation + SH
│   ├── models/                       ← encodeurs + modèle contrastif + Trainer
│   ├── inference/                    ← Gallery + HRTFPredictor
│   ├── evaluation/                   ← RecallCallback + ValidationReport + LSD
│   ├── synthesis/
│   └── analysis/
├── checkpoints/                      ← poids + configs + mlruns.db (non versionnés)
├── sound/                            ← fichiers générés (non versionnés)
├── docs/
│   ├── hrtf_interpolation.tex
│   └── hrtf_crossmodal.tex           ← architecture HRTF cross-modal (LaTeX)
├── notebooks/
│   ├── static HRTF Demo.ipynb
│   ├── DynamicConvolver_viz.ipynb
│   ├── DynamicSoundscape_viz.ipynb
│   ├── HRTFGen.ipynb
│   ├── SegmentEngine_Viz.ipynb
│   ├── traj_viz.ipynb
│   ├── Build_Tchaikovsky.ipynb       ← reconstruction scène orchestrale
│   ├── Build_Harmo_Spher.ipynb       ← projection HRTF → harmoniques sphériques
│   ├── Prepro_dataset.ipynb          ← preprocessing du dataset cross-modal
│   ├── Build_EAR_Encodeur.ipynb      ← construction et test de l'EarEncoder
│   ├── Build_HRTF_Encodeur.ipynb     ← construction et test du HRTFEncoder
│   ├── Train_constrative_model.ipynb ← entraînement complet (3 phases, MLflow)
│   ├── Plot_viz_model.ipynb          ← visualisation de l'espace latent
│   └── Build_full_pipeline_inference.ipynb ← Gallery + HRTFPredictor end-to-end
├── pyproject.toml
└── Requirements.txt
```

---

## Dépendances

**Spatialisation audio (module de base) :**

| Package | Usage |
|---|---|
| `numpy` | Calcul numérique |
| `scipy` | FFT, convolution, géométrie (ConvexHull), fenêtres WOLA |
| `soundfile` | Lecture/écriture WAV/FLAC |
| `netCDF4` | Lecture/écriture fichiers SOFA |
| `matplotlib` | Visualisations |
| `librosa` | Rééchantillonnage |
| `jupyter` | Notebooks interactifs |

**Module cross-modal (personnalisation HRTF) :**

| Package | Usage |
|---|---|
| `tensorflow` | EarEncoder, HRTFEncoder, ContrastiveModel, entraînement |
| `scikit-learn` | `train_test_split`, métriques de classification |
| `mlflow` | Versioning des expériences, logging des métriques et artefacts |
| `tensorboard` | Visualisation live des courbes d'entraînement et histogrammes |
| `tqdm` | Barre de progression (`TqdmCallback`) |
| `Pillow` | Chargement et redimensionnement des photos d'oreilles |

```bash
pip install -r Requirements.txt
```

---

## Références

Base de données IRCAM LISTEN :
> UMR 9912 - STMS - IRCAM/CNRS/UPMC. *LISTEN HRTF Database* — Olivier Warusfel.

Dataset The Spheres :
> Gaultier, C. et al. *The Spheres: A Multichannel Database of Orchestral Music*. 2024.

Licence IRCAM : utilisation libre à des fins éducatives, de recherche ou commerciales.
