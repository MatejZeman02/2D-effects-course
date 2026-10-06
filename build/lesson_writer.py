"""Writes a lesson's notebook and its solved copy, `<name>_solution.ipynb`, from the cells a builder made.

A builder's `code(text, solve=[(stub, solution), ...])` keeps the solved
source of a task cell, *text* with each stub replaced, under the cell's
`solution` key. The lesson gets the cell as the student sees
it, the solved copy gets the solution in its place, so the author can run a
whole lesson with every task done. Neither file keeps the key. The solved
copies are in `.gitignore`, since a student who wants one has the 🔑 boxes.
"""

import json
from pathlib import Path

METADATA = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}


def lines(text):
    """A cell's source as the notebook format keeps it, a list of lines with their newlines."""
    return text.strip("\n").splitlines(keepends=True)


def solved(text, solve):
    """The lines of *text* with each `(stub, solution)` of *solve* replaced, so a stub a later edit
    renamed fails the build instead of leaving a task unsolved in the solved copy."""
    for stub, solution in solve:
        if stub not in text:
            raise ValueError(f"the cell holds no stub {stub.strip()[:60]!r}")
        text = text.replace(stub, solution)
    return lines(text)


def solution_path(out):
    """Where the solved copy of the lesson at *out* goes, beside it."""
    return out.with_name(out.stem + "_solution.ipynb")


def notebook(cells, prefix, solved=False):
    """The notebook of *cells*, with ids `<prefix>-NN`, the solutions swapped in when *solved*."""
    written = []
    for i, cell in enumerate(cells):
        cell = dict(cell)
        solution = cell.pop("solution", None)
        if solved and solution is not None:
            cell["source"] = solution
        cell["id"] = f"{prefix}-{i:02d}"
        written.append(cell)
    if solved:
        title = written[0]
        title["source"] = [title["source"][0].rstrip("\n") + " (řešení)\n"] + title["source"][1:]
    return {"cells": written, "metadata": METADATA, "nbformat": 4, "nbformat_minor": 5}


def write(cells, out, prefix):
    """Writes the lesson to *out* and its solved copy beside it, and says how many cells each holds."""
    for path, solved in ((Path(out), False), (solution_path(Path(out)), True)):
        path.write_text(json.dumps(notebook(cells, prefix, solved), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(path, len(cells), "cells")
