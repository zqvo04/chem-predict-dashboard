"""STEP 7 tests: wide library canonicalisation + dedup (offline, mocked download)."""
import gzip

from src.data import library as lib


def test_load_library_canonicalises_and_dedups(tmp_path, monkeypatch):
    # Two spellings of ethanol + benzene + one unparseable -> 2 unique canonical rows.
    csv = b"smiles,foo\nOCC,1\nCCO,1\nc1ccccc1,1\nnot_a_smiles,1\n"
    monkeypatch.setattr(lib, "CACHE", tmp_path / "library.parquet")
    monkeypatch.setattr(lib, "_download", lambda name: gzip.compress(csv))
    out = lib.load_library(use_cache=False)
    assert set(out["smi"]) == {"CCO", "c1ccccc1"}        # deduped + canonical, bad dropped
    assert out["smi"].is_unique
    assert (tmp_path / "library.parquet").exists()        # cached


def test_load_library_uses_cache(tmp_path, monkeypatch):
    csv = b"smiles,foo\nCCO,1\n"
    monkeypatch.setattr(lib, "CACHE", tmp_path / "library.parquet")
    monkeypatch.setattr(lib, "_download", lambda name: gzip.compress(csv))
    lib.load_library(use_cache=False)                     # writes cache
    monkeypatch.setattr(lib, "_download",
                        lambda name: (_ for _ in ()).throw(AssertionError("re-downloaded")))
    assert list(lib.load_library(use_cache=True)["smi"]) == ["CCO"]
