"""Download a published Release asset and verify it against the manifest."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe", nargs="?", help="e.g. llada-base/python-single")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--tag", help="Published GitHub Release tag, e.g. v0.1.0")
    args = parser.parse_args()
    weights = json.loads((ROOT / "checkpoints/manifest.json").read_text())["weights"]
    if args.list:
        for item in weights:
            print(item["id"])
        return
    selected = next((item for item in weights if item["id"] == args.probe), None)
    if selected is None:
        parser.error("Select a probe listed by --list.")
    if not args.tag:
        parser.error("--tag is required; use a published release tag.")
    destination = ROOT / selected["path"]
    if (
        destination.exists()
        and hashlib.sha256(destination.read_bytes()).hexdigest() == selected["sha256"]
    ):
        print(destination)
        return
    url = f"https://github.com/Hsu1023/PILL/releases/download/{args.tag}/{selected['asset']}"
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
