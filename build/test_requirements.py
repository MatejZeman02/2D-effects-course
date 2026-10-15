"""The notebooks' NumPy is the one the Krita plugin carries, so an effect behaves the same in both.

    python3 -m pytest build/test_requirements.py
"""
import re
import sys
from pathlib import Path

BUILD = Path(__file__).resolve().parent
sys.path.insert(0, str(BUILD))

import krita_template  # noqa: E402


def test_requirements_pin_the_plugins_numpy():
    text = (BUILD.parent / "requirements.txt").read_text(encoding="utf-8")
    pins = re.findall(r"^numpy==([0-9.]+)", text, re.MULTILINE)
    assert pins == [krita_template.NUMPY]
