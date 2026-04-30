#!/usr/bin/env python3
"""Download the IRCAM SOFA file to the current folder (OOP version).

Usage:
    python "download database.py" --id-irc 1002
"""

import argparse
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE_URL = "http://bili2.ircam.fr:80/SimpleFreeFieldHRIR/LISTEN/COMPENSATED"
DEFAULT_ID_IRC = 1002
DEFAULT_SAMPLE_RATE = 44100
OUTPUT_DIR = Path(".")
CHUNK_SIZE = 8192


class SofaDownloader:
    """Handle SOFA file download from an HTTP URL."""

    def __init__(
        self,
        id_irc: int,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        output_dir: Path = OUTPUT_DIR,
        chunk_size: int = CHUNK_SIZE,
        base_url: str = BASE_URL,
    ) -> None:
        if id_irc <= 0:
            raise ValueError("id_irc must be a positive integer")
        self.id_irc = id_irc
        self.sample_rate = sample_rate
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.base_url = base_url

    @property
    def filename(self) -> str:
        return f"IRC_{self.id_irc}_C_{self.sample_rate}.sofa"

    @property
    def url(self) -> str:
        return f"{self.base_url}/{self.sample_rate}/{self.filename}"

    @property
    def destination(self) -> Path:
        return self.output_dir / self.filename

    @staticmethod
    def _request_headers() -> dict[str, str]:
        # Some servers block the default Python user agent and return HTTP 403.
        return {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "*/*",
            "Connection": "keep-alive",
        }

    def _candidate_urls(self) -> list[str]:
        primary = self.url
        if primary.startswith("http://"):
            return [primary, primary.replace("http://", "https://", 1)]
        if primary.startswith("https://"):
            return [primary, primary.replace("https://", "http://", 1)]
        return [primary]

    def download(self) -> Path:
        """Stream the file from URL and save it to destination."""
        destination = self.destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp_destination = destination.with_suffix(destination.suffix + ".part")

        last_error: Exception | None = None
        for candidate_url in self._candidate_urls():
            request = Request(candidate_url, headers=self._request_headers())
            try:
                with urlopen(request, timeout=30) as response, tmp_destination.open("wb") as target:
                    while True:
                        chunk = response.read(self.chunk_size)
                        if not chunk:
                            break
                        target.write(chunk)
                tmp_destination.replace(destination)
                return destination
            except (HTTPError, URLError, TimeoutError) as error:
                last_error = error
                if tmp_destination.exists():
                    tmp_destination.unlink()

        raise RuntimeError(
            f"Unable to download {self.filename}. Last error: {last_error}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download an IRCAM SOFA file by IRC id.")
    parser.add_argument(
        "--id-irc",
        type=int,
        default=DEFAULT_ID_IRC,
        help=f"IRC id to download (default: {DEFAULT_ID_IRC})",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help=f"Sample rate used in URL and filename (default: {DEFAULT_SAMPLE_RATE})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    downloader = SofaDownloader(
        id_irc=args.id_irc,
        sample_rate=args.sample_rate,
        chunk_size=CHUNK_SIZE,
    )
    print(f"Downloading: {downloader.url}")
    saved_path = downloader.download()
    print(f"Saved to: {saved_path.resolve()}")


if __name__ == "__main__":
    main()
