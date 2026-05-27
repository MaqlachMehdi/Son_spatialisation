from pathlib import Path

import numpy as np
from PIL import Image


class EarProcessor:
    """Charge et normalise les photos d'oreilles gauche et droite.

    Output : deux np.ndarray de shape [img_size, img_size, 3], float32 dans [0, 1]
    """

    def __init__(self, img_size: int = 224) -> None:
        self.img_size = img_size

    def load(self, img_L: Path, img_R: Path) -> tuple[np.ndarray, np.ndarray]:
        return self._load_one(img_L), self._load_one(img_R)

    def _load_one(self, path: Path) -> np.ndarray:
        img = Image.open(str(path)).convert('RGB')
        img = img.resize((self.img_size, self.img_size), Image.LANCZOS)
        return np.array(img, dtype=np.float32) / 255.0
