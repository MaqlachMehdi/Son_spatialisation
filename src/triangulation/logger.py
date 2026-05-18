"""
logger.py
---------
Logger sémantique pour le suivi du pipeline de localisation.
"""

from __future__ import annotations

import ctypes
import logging
import sys
import time

import numpy as np


# ── Détection du support ANSI ─────────────────────────────────────────────────

def _enable_ansi() -> bool:
    """
    Retourne True si le flux de sortie accepte les codes ANSI.

    Ordre de priorité :
    1. Jupyter  → toujours True  (les cellules rendent l'ANSI)
    2. tty Unix → True si stderr est un terminal
    3. Windows  → active Virtual Terminal Processing sur la console, ou False
    """
    try:
        from IPython import get_ipython  # type: ignore
        if get_ipython() is not None:
            return True
    except ImportError:
        pass

    stream = sys.stderr
    if not (hasattr(stream, "isatty") and stream.isatty()):
        return False

    if sys.platform == "win32":
        try:
            STD_ERROR_HANDLE                = -12
            ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
            handle = ctypes.windll.kernel32.GetStdHandle(STD_ERROR_HANDLE)
            mode   = ctypes.c_ulong()
            ctypes.windll.kernel32.GetConsoleMode(handle, ctypes.byref(mode))
            ctypes.windll.kernel32.SetConsoleMode(
                handle, mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING
            )
        except Exception:
            return False

    return True


_USE_ANSI = _enable_ansi()
_c = (lambda code: code) if _USE_ANSI else (lambda code: "")

_RESET  = _c("\033[0m")
_BOLD   = _c("\033[1m")
_BLUE   = _c("\033[34m")
_GREEN  = _c("\033[32m")
_YELLOW = _c("\033[33m")
_RED    = _c("\033[31m")
_CYAN   = _c("\033[36m")
_GREY   = _c("\033[90m")


# ── Formatter ─────────────────────────────────────────────────────────────────

class _TriangulationFormatter(logging.Formatter):
    _COLORS = {
        logging.DEBUG:   _GREY,
        logging.INFO:    _RESET,
        logging.WARNING: _YELLOW,
        logging.ERROR:   _RED,
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self._COLORS.get(record.levelno, _RESET)
        return f"{color}{super().format(record)}{_RESET}"


# ── Logger sémantique ─────────────────────────────────────────────────────────

class TriangulationLogger:
    """
    Logger sémantique pour le pipeline de localisation.

    Méthodes
    --------
    start(title)                          → bandeau de début
    step(n, total, name)                  → [n/total] Nom de l'étape
    matrix_info(name, M)                  → shape + stats (niveau DEBUG)
    scalar(label, value, unit)            → valeur scalaire indentée
    eigenvalues(Lambda)                   → tableau formaté + variance expliquée
    check(label, ok, detail)              → ✓ ou ⚠ avec détail optionnel
    positions_table(mic_ids, labels, X)   → tableau des coordonnées cartésiennes
    warn(msg)                             → avertissement visible
    done()                                → bandeau de fin avec durée
    """

    def __init__(self, name: str = "triangulation", level: int = logging.INFO) -> None:
        self._log = logging.getLogger(name)
        if not self._log.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(_TriangulationFormatter("%(message)s"))
            self._log.addHandler(handler)
        self._log.setLevel(level)
        self._log.propagate = False
        self._t0: float | None = None

    # ── Contrôle du niveau ────────────────────────────────────────────────────

    def set_debug(self) -> None:
        self._log.setLevel(logging.DEBUG)

    def set_info(self) -> None:
        self._log.setLevel(logging.INFO)

    # ── Méthodes sémantiques ──────────────────────────────────────────────────

    def start(self, title: str) -> None:
        self._t0 = time.perf_counter()
        self._log.info(
            f"\n{_BOLD}{'═' * 58}{_RESET}\n"
            f"{_BOLD}  {title}{_RESET}\n"
            f"{_BOLD}{'═' * 58}{_RESET}"
        )

    def step(self, n: int, total: int, name: str) -> None:
        self._log.info(f"\n{_CYAN}[{n}/{total}]{_RESET} {_BOLD}{name}{_RESET}")

    def matrix_info(self, name: str, M: np.ndarray) -> None:
        if not self._log.isEnabledFor(logging.DEBUG):
            return
        off = M[~np.eye(M.shape[0], dtype=bool)] if M.ndim == 2 else M
        off = off[off != 0]
        if off.size == 0:
            return
        self._log.debug(
            f"       {name:<14s} shape={M.shape}  "
            f"min={off.min():.3f}  max={off.max():.3f}  mean={off.mean():.3f}"
        )

    def scalar(self, label: str, value: float, unit: str = "") -> None:
        self._log.info(f"       {label:<35s} {value:.6g} {unit}")

    def eigenvalues(self, Lambda: np.ndarray) -> None:
        total    = float(Lambda.sum()) + 1e-12
        ratio_3d = Lambda[:3].sum() / total * 100

        lines = ["\n       Valeurs propres de B"]
        for k, lk in enumerate(Lambda[:6]):
            marker  = f"{_GREEN}◆{_RESET}" if k < 3 else f"{_GREY}·{_RESET}"
            bar_len = int(lk / (Lambda[0] + 1e-12) * 24)
            bar     = f"{_BLUE}{'█' * bar_len}{_GREY}{'░' * (24 - bar_len)}{_RESET}"
            lines.append(f"       {marker} λ{k + 1:2d} = {lk:>12.3f}  {bar}")
        if len(Lambda) > 6:
            rest = Lambda[6:]
            lines.append(
                f"       {_GREY}  λ7..{len(Lambda)}  max={rest.max():.3e}  "
                f"(bruit numérique){_RESET}"
            )
        lines.append(
            f"\n       Variance expliquée (3 axes) : "
            f"{_GREEN}{_BOLD}{ratio_3d:.2f}%{_RESET}"
        )
        self._log.info("\n".join(lines))

    def check(self, label: str, ok: bool, detail: str = "") -> None:
        icon  = f"{_GREEN}✓{_RESET}" if ok else f"{_YELLOW}⚠{_RESET}"
        level = logging.INFO if ok else logging.WARNING
        msg   = f"       {icon}  {label}"
        if detail:
            msg += f"  {_GREY}({detail}){_RESET}"
        self._log.log(level, msg)

    def positions_table(
        self,
        mic_ids: list[str],
        instrument_labels: list[str],
        X: np.ndarray,
    ) -> None:
        header = (
            f"       {'Mic':<10} {'Instrument':<15}"
            f" {'X (m)':>9} {'Y (m)':>9} {'Z (m)':>9}"
        )
        sep  = f"       {'─' * 10} {'─' * 15} {'─' * 9} {'─' * 9} {'─' * 9}"
        rows = ["\n       Coordonnées cartésiennes (avant conversion sphérique)", header, sep]
        for i, (mic_id, instr) in enumerate(zip(mic_ids, instrument_labels)):
            rows.append(
                f"       {mic_id:<10} {instr:<15}"
                f" {X[i, 0]:>9.3f} {X[i, 1]:>9.3f} {X[i, 2]:>9.3f}"
            )
        self._log.info("\n".join(rows))

    def info(self, msg: str) -> None:
        self._log.info(f"       {msg}")

    def warn(self, msg: str) -> None:
        self._log.warning(f"       {_YELLOW}⚠  {msg}{_RESET}")

    def done(self) -> None:
        elapsed = (time.perf_counter() - self._t0) * 1000 if self._t0 else 0.0
        self._log.info(
            f"\n{_GREEN}{'─' * 58}{_RESET}\n"
            f"{_GREEN}  Pipeline terminé en {elapsed:.1f} ms{_RESET}\n"
            f"{_GREEN}{'─' * 58}{_RESET}\n"
        )


# Instance globale du module — importée par tous les autres fichiers
logger = TriangulationLogger()
