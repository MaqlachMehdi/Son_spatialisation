#!/usr/bin/env python3
"""Download the IRCAM SOFA file to the current folder.

Usage:
    python "download database.py"
"""

from pathlib import Path
from urllib.request import urlopen

SOFA_URL = "http://bili2.ircam.fr:80/SimpleFreeFieldHRIR/LISTEN/COMPENSATED/44100/IRC_1002_C_44100.sofa"
OUTPUT_FILE = Path("IRC_1002_C_44100.sofa")
CHUNK_SIZE = 8192


def download_file(url: str, destination: Path) -> None:
    """Stream a file from URL to destination on disk."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    with urlopen(url) as response, destination.open("wb") as target:
        while True:
            chunk = response.read(CHUNK_SIZE)
            if not chunk:
                break
            target.write(chunk)


def main() -> None:
    print(f"Downloading: {SOFA_URL}")
    download_file(SOFA_URL, OUTPUT_FILE)
    print(f"Saved to: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
