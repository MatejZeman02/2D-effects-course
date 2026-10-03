"""The zip the builder packs, read the way Krita's Import Python Plugin from File reads it."""

import importlib.util
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("krita_template", ROOT / "build" / "krita_template.py")
krita_template = importlib.util.module_from_spec(spec)
spec.loader.exec_module(krita_template)


def packed(tmp_path, monkeypatch) -> list[str]:
    monkeypatch.setattr(krita_template, "DIST", tmp_path)
    with zipfile.ZipFile(krita_template.pack("linux")) as z:
        return z.namelist()


def test_krita_finds_the_plugin_folder(tmp_path, monkeypatch):
    # Krita's plugin_importer.py looks for an entry "pga_filter/" with
    # "pga_filter/__init__.py" beside it, and says "No plugins found in archive" without one.
    names = packed(tmp_path, monkeypatch)
    assert "pga_filter.desktop" in names
    assert "pga_filter/" in names
    assert "pga_filter/__init__.py" in names


def test_every_folder_has_an_entry_before_its_files(tmp_path, monkeypatch):
    names = packed(tmp_path, monkeypatch)
    seen = set()
    for name in names:
        parents = name.rstrip("/").split("/")[:-1]
        for depth in range(1, len(parents) + 1):
            assert "/".join(parents[:depth]) + "/" in seen, name
        if name.endswith("/"):
            seen.add(name)
