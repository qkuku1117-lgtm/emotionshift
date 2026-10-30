"""Download the official EmoWOZ release without touching existing datasets."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "emowoz"
BASE = "https://zenodo.org/records/6506504/files/"
FILES = ("emowoz-multiwoz.json", "emowoz-dialmage.json", "data-split.json")


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name in FILES:
        url = BASE + name + "?download=1"
        with urlopen(url, timeout=300) as response:
            content = response.read()
        json.loads(content)  # Validate before replacing a local file.
        (RAW / name).write_bytes(content)
        manifest[name] = {"url": url, "sha256": hashlib.sha256(content).hexdigest()}
    (RAW / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Downloaded official EmoWOZ release to", RAW)


if __name__ == "__main__":
    main()
