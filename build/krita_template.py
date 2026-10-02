"""Packs the Krita template into one zip per system, each with its own NumPy.

    python build/krita_template.py --download            # fetch both NumPy wheels from PyPI, then pack
    python build/krita_template.py --linux-numpy DIR     # use a NumPy already installed for Python 3.13

Krita 6 ships Python 3.13 and no NumPy, so the plugin carries NumPy in
pga_filter/vendor/<system>, which its __init__.py puts on the path. The
wheels are unpacked there, and krita/dist/ gets pga_filter-linux.zip and
pga_filter-windows.zip for every system that has one. Krita imports such a zip
through Tools > Scripts > Import Python Plugin from File.

A downloaded wheel is kept in build/.wheels, so a second run needs no network.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KRITA = ROOT / "krita"
PLUGIN = KRITA / "pga_filter"
DESKTOP = KRITA / "pga_filter.desktop"
DIST = KRITA / "dist"
WHEELS = Path(__file__).resolve().parent / ".wheels"

NUMPY = "2.5.1"
PYTHON = "3.13"
PLATFORMS = {"linux": "manylinux_2_28_x86_64", "windows": "win_amd64"}
SKIP = {"__pycache__", "vendor"}


def download(system: str) -> Path:
    """The NumPy wheel for *system*, fetched from PyPI unless it is already in build/.wheels."""
    WHEELS.mkdir(exist_ok=True)
    marker = "win_amd64" if system == "windows" else "manylinux"

    def found() -> list[Path]:
        return sorted(w for w in WHEELS.glob(f"numpy-{NUMPY}-cp313-cp313-*.whl") if marker in w.name)

    if not found():
        subprocess.run([sys.executable, "-m", "pip", "download", f"numpy=={NUMPY}", "--no-deps",
                        "--only-binary=:all:", "--platform", PLATFORMS[system], "--python-version",
                        PYTHON, "--implementation", "cp", "--abi", "cp313", "-d", str(WHEELS)], check=True)
    return found()[-1]


def unpack_wheel(wheel: Path, target: Path) -> None:
    """numpy/ and numpy.libs/ from *wheel* into *target*, without NumPy's own test suites."""
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)
    with zipfile.ZipFile(wheel) as z:
        for name in z.namelist():
            if name.startswith(("numpy/", "numpy.libs/")) and "/tests/" not in name:
                z.extract(name, target)


def copy_installed(site: Path, target: Path) -> None:
    """numpy/ and numpy.libs/ from an installed site-packages folder into *target*."""
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)
    for name in ("numpy", "numpy.libs"):
        if (site / name).is_dir():
            shutil.copytree(site / name, target / name, ignore=shutil.ignore_patterns("__pycache__", "tests"))


def pack(system: str) -> Path:
    """krita/dist/pga_filter-<system>.zip: the .desktop file, the plugin and this system's NumPy."""
    DIST.mkdir(exist_ok=True)
    out = DIST / f"pga_filter-{system}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(DESKTOP, DESKTOP.name)
        for path in sorted(PLUGIN.rglob("*")):
            rel = path.relative_to(KRITA)
            parts = path.relative_to(PLUGIN).parts
            if path.is_dir() or parts[0] in SKIP or "__pycache__" in parts:
                continue
            z.write(path, rel.as_posix())
        vendor = PLUGIN / "vendor" / system
        for path in sorted(vendor.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                z.write(path, path.relative_to(KRITA).as_posix())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--download", action="store_true", help="fetch the NumPy wheels from PyPI")
    parser.add_argument("--linux-numpy", type=Path, help="site-packages holding NumPy for Python 3.13 on Linux")
    args = parser.parse_args()
    if args.linux_numpy:
        copy_installed(args.linux_numpy, PLUGIN / "vendor" / "linux")
    if args.download:
        for system in PLATFORMS:
            unpack_wheel(download(system), PLUGIN / "vendor" / system)
    for system in PLATFORMS:
        if (PLUGIN / "vendor" / system / "numpy").is_dir():
            out = pack(system)
            print(out, f"{out.stat().st_size / 1e6:.1f} MB")
        else:
            print(f"no NumPy for {system} yet, run with --download", file=sys.stderr)


if __name__ == "__main__":
    main()
