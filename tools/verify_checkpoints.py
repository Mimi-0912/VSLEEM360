"""Verify externally downloaded checkpoints without loading model code."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--task", choices=("vpr", "vsr", "all"), default="all")
    args = parser.parse_args()
    folders = {"vpr": "VPR/weights", "vsr": "SR/weights"}
    failed = False
    for task, relative in folders.items():
        if args.task not in (task, "all"):
            continue
        folder = args.root / relative
        records = json.loads((folder / "checkpoint_manifest.json").read_text())["checkpoints"]
        for record in records:
            path = folder / record["checkpoint"]
            if not path.is_file():
                print(f"MISSING {relative}/{path.name}")
                failed = True
                continue
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 << 20), b""):
                    digest.update(block)
            valid = (path.stat().st_size == record["bytes"]
                     and digest.hexdigest() == record["sha256"])
            print(f"{'OK' if valid else 'MISMATCH'} {relative}/{path.name}")
            failed |= not valid
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
