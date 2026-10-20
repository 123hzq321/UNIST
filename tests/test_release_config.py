import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_assets_are_version_addressed():
    records = json.loads((ROOT / "configs/joint_sources_locked.json").read_text(encoding="utf-8-sig"))
    assert len(records) == 3
    assert sum(rec["expected_cells"] for rec in records) == 565598
    for rec in records:
        assert rec["url"] == f'https://datasets.cellxgene.cziscience.com/{rec["version_id"]}.h5ad'
        assert len(rec["sha256"]) == 64
        assert "local_path" not in rec
        assert "declared_donors" not in rec


def test_download_digest_reads_full_file(tmp_path):
    spec = importlib.util.spec_from_file_location("locked_download", ROOT / "scripts/fetch_locked_sources.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "payload"
    path.write_bytes(b"abc")
    assert module.sha256(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
