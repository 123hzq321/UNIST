"""Fetch or validate the exact public source files used in the joint experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import h5py
import requests
from anndata.io import read_elem


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--download", action="store_true", help="Download missing files (about 5.4 GB).")
    args = parser.parse_args()
    root = args.root.resolve()
    configs = Path(__file__).resolve().parents[1] / "configs"
    records = json.loads((configs / "joint_sources_locked.json").read_text(encoding="utf-8-sig"))
    (root / "raw").mkdir(parents=True, exist_ok=True)
    for rec in records:
        target = root / "raw" / (rec["cancer"] + ".h5ad")
        if not target.exists():
            if not args.download:
                raise FileNotFoundError(f"Missing {target}; use --download to fetch the locked source.")
            partial = target.with_suffix(".h5ad.part")
            if partial.exists():
                raise FileExistsError(f"Inspect the previous partial download before retrying: {partial}")
            with requests.get(rec["url"], stream=True, timeout=(30, 180)) as response:
                response.raise_for_status()
                with partial.open("xb") as stream:
                    for block in response.iter_content(8 * 1024 * 1024):
                        stream.write(block)
            if partial.stat().st_size != rec["expected_bytes"] or sha256(partial) != rec["sha256"]:
                raise ValueError(f"Downloaded file failed the locked size/hash check: {partial}")
            partial.rename(target)
        if target.stat().st_size != rec["expected_bytes"] or sha256(target) != rec["sha256"]:
            raise ValueError(f"Existing file is not the locked version: {target}")
        with h5py.File(target, "r") as handle:
            obs = read_elem(handle["obs"])
            if len(obs) != rec["expected_cells"]:
                raise ValueError(f"Cell count mismatch: {target}")
            rec["declared_donors"] = sorted(obs["donor_id"].astype(str).unique().tolist())
        rec.update(local_path=str(target), status="downloaded_structurally_validated")
        print("VALIDATED", rec["cancer"], flush=True)
    crosswalk = root / "label_crosswalk.json"
    expected = json.loads((configs / "label_crosswalk.json").read_text(encoding="utf-8"))
    if crosswalk.exists() and json.loads(crosswalk.read_text(encoding="utf-8")) != expected:
        raise ValueError("Existing label crosswalk differs from the frozen study configuration.")
    if not crosswalk.exists():
        shutil.copyfile(configs / "label_crosswalk.json", crosswalk)
    (root / "source_manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
