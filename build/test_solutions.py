"""Every task cell of every lesson has a solution, so the solved copies run with nothing left to do.

    python3 -m pytest build/test_solutions.py
"""
import importlib
import sys
from pathlib import Path

import pytest

BUILD = Path(__file__).resolve().parent
sys.path.insert(0, str(BUILD))

import lesson_writer  # noqa: E402

LESSONS = sorted(path.stem for path in BUILD.glob("lesson_*_notebook.py"))


@pytest.mark.parametrize("name", LESSONS)
def test_every_stub_has_a_solution_that_leaves_none(name):
    nb = importlib.import_module(name)
    stubs = [cell for cell in nb.cells if cell["cell_type"] == "code" and "TODO" in "".join(cell["source"])]
    assert stubs, "a lesson with no task cell"
    for cell in stubs:
        assert "solution" in cell, "".join(cell["source"])[:200]
        solved = "".join(cell["solution"])
        assert "TODO" not in solved, solved[:200]
        if not solved.startswith("%"):
            compile(solved, name, "exec")


@pytest.mark.parametrize("name", LESSONS)
def test_the_solved_copy_differs_only_in_its_title_and_its_task_cells(name):
    nb = importlib.import_module(name)
    given = lesson_writer.notebook(nb.cells, "x")
    solved = lesson_writer.notebook(nb.cells, "x", solved=True)
    assert solved["cells"][0]["source"][0].rstrip().endswith("(řešení)")
    changed = [i for i, (a, b) in enumerate(zip(given["cells"], solved["cells"])) if a["source"] != b["source"]]
    assert changed[1:] == [i for i, cell in enumerate(nb.cells) if "solution" in cell]
    assert all("solution" not in cell for cell in given["cells"] + solved["cells"])
