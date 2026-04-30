#!/usr/bin/env python3
"""Download the IRCAM SOFA file to the current folder (OOP version).

Usage:
    python "download database.py"
"""

from pathlib import Path
from urllib.request import urlopen

SOFA_URL = "http://bili2.ircam.fr:80/SimpleFreeFieldHRIR/LISTEN/COMPENSATED/44100/IRC_1002_C_44100.sofa"
OUTPUT_FILE = Path("IRC_1002_C_44100.sofa")
CHUNK_SIZE = 8192


class SofaDownloader:
    """Handle SOFA file download from an HTTP URL."""

    def __init__(self, url: str, destination: Path, chunk_size: int = 8192) -> None:
        self.url = url
        self.destination = destination
        self.chunk_size = chunk_size

    def download(self) -> Path:
        """Stream the file from URL and save it to destination."""
        self.destination.parent.mkdir(parents=True, exist_ok=True)

        with urlopen(self.url) as response, self.destination.open("wb") as target:
            while True:
                chunk = response.read(self.chunk_size)
                if not chunk:
                    break
                target.write(chunk)

        return self.destination


def main() -> None:
    downloader = SofaDownloader(
        url=SOFA_URL,
        destination=OUTPUT_FILE,
        chunk_size=CHUNK_SIZE,
    )
    print(f"Downloading: {downloader.url}")
    saved_path = downloader.download()
    print(f"Saved to: {saved_path.resolve()}")


if __name__ == "__main__":
    main()
