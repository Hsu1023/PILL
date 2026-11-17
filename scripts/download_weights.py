"""Download a repository checkpoint and verify it against the manifest."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request
import urllib.parse

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe", nargs="?", help="e.g. llada-base/python-single")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--ref", default="main", help="Git branch, tag, or commit (default: main)")
    args = parser.parse_args()
    weights = json.loads((ROOT / "checkpoints/manifest.json").read_text())["weights"]
    if args.list:
        for item in weights:
            print(item["id"])
        return
    selected = next((item for item in weights if item["id"] == args.probe), None)
    if selected is None:
        parser.error("Select a probe listed by --list.")
    destination = ROOT / selected["path"]
    if (
        destination.exists()
        and hashlib.sha256(destination.read_bytes()).hexdigest() == selected["sha256"]
    ):
        print(destination)
        return
    ref = urllib.parse.quote(args.ref, safe="")
    path = urllib.parse.quote(selected["path"], safe="/")
    url = f"https://raw.githubusercontent.com/Hsu1023/PILL/{ref}/{path}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            digest = hashlib.sha256()
            with urllib.request.urlopen(url, timeout=60) as response:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    handle.write(block)
                    digest.update(block)
        if digest.hexdigest() != selected["sha256"]:
            raise ValueError("Downloaded weight checksum does not match the manifest.")
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(destination)


if __name__ == "__main__":
    main()
