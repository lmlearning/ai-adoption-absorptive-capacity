"""Fetch the pinned official survey CSV and verify it before replacement."""
import argparse
import hashlib
from pathlib import Path
import tempfile
from urllib.request import urlopen

SOURCE_COMMIT = "32a114542da67e3759479637343718502742adfd"
SOURCE_URL = f"https://media.githubusercontent.com/media/StackExchange/Survey/{SOURCE_COMMIT}/packages/archive/2025/results.csv"
SOURCE_SHA256 = "2d1f65308877282edfb4470520eabbc08cb499118432a3dcec6a66c086aa2baa"


def checksum(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(destination, url=SOURCE_URL, expected_sha256=SOURCE_SHA256):
    destination = Path(destination)
    if destination.exists():
        if checksum(destination) != expected_sha256:
            raise ValueError(f"{destination} exists but its checksum differs; preserved without replacement")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".survey-", suffix=".tmp", delete=False) as target:
            temporary = Path(target.name)
            with urlopen(url, timeout=60) as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    target.write(chunk)
        if checksum(temporary) != expected_sha256:
            raise ValueError("Downloaded survey checksum does not match the pinned official data")
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("stackoverflow_2025_survey.csv"))
    args = parser.parse_args()
    try:
        path = download(args.output)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Download failed: {error}\n")
    print(f"Verified {path} (SHA-256 {SOURCE_SHA256})")


if __name__ == "__main__":
    main()
