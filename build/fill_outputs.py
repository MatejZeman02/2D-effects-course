"""Fills each lesson notebook with the outputs of its solved copy, so a student sees how every cell
should come out before writing the tasks, and after running a task sees whether it changed.

Run it after the builders, which write the notebooks empty, with Sara running:

    python build/fill_outputs.py          # every lesson
    python build/fill_outputs.py 4 5      # lessons 4 and 5

It runs `lesson_NN_*_solution.ipynb` top to bottom in the course folder, then copies each cell's
outputs into the same cell of `lesson_NN_*.ipynb`, whose task cells keep their stubs. A saved widget
needs its kernel to draw, and VS Code keeps no widget state at all, so each widget output becomes the
picture and the caption it showed last. A large picture, a photo or a spectrum, is kept as a JPEG.
A lesson with a failing cell is left as it was.
"""
import base64
import io
import json
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from PIL import Image

HERE = Path(__file__).resolve().parent
COURSE = HERE.parent
sys.path.insert(0, str(HERE))
from lesson_writer import solution_path  # noqa: E402

WIDGET = "application/vnd.jupyter.widget-view+json"
STATE = "application/vnd.jupyter.widget-state+json"
BOXES = {"VBoxModel", "HBoxModel", "BoxModel", "GridBoxModel"}
LARGE = 200_000        # bytes of PNG past which a JPEG is tried, for photos


def picture(png):
    """A display of the PNG bytes *png*, as a JPEG when that is under half the size."""
    if len(png) > LARGE:
        image = Image.open(io.BytesIO(png)).convert("RGBA")
        flat = Image.new("RGB", image.size, "white")
        flat.paste(image, mask=image.getchannel("A"))
        jpeg = io.BytesIO()
        flat.save(jpeg, "JPEG", quality=85)
        if jpeg.tell() < len(png) / 2:
            return nbformat.v4.new_output("display_data", data={"image/jpeg": base64.b64encode(jpeg.getvalue()).decode()})
    return nbformat.v4.new_output("display_data", data={"image/png": base64.b64encode(png).decode()})


def still(model_id, models):
    """The outputs that show what the widget *model_id* showed: its pictures, captions and labels, no controls."""
    model = models.get(model_id.removeprefix("IPY_MODEL_"))
    if model is None:
        return []
    kind, state = model["model_name"], model["state"]
    if kind in BOXES:
        return [shown for child in state.get("children", []) for shown in still(child, models)]
    if kind == "ImageModel":
        value = [b["data"] for b in model.get("buffers", []) if b["path"] == ["value"]]
        return [picture(base64.b64decode(value[0]))] if value else []
    if kind == "HTMLModel" and state.get("value"):
        return [nbformat.v4.new_output("display_data", data={"text/html": state["value"]})]
    if kind == "LabelModel" and state.get("value"):
        return [nbformat.v4.new_output("display_data", data={"text/plain": state["value"]})]
    return []


def fill(lesson):
    """Runs *lesson*'s solved copy and writes its outputs into *lesson*, or raises at the first failing cell."""
    ran = nbformat.read(solution_path(lesson), as_version=4)
    NotebookClient(ran, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(COURSE)}},
                   store_widget_state=True).execute()
    models = ran.metadata.get("widgets", {}).get(STATE, {}).get("state", {})
    notebook = json.loads(lesson.read_text(encoding="utf-8"))
    if len(notebook["cells"]) != len(ran.cells):
        raise ValueError(f"{lesson.name} and its solved copy differ in cells, build them again")
    for cell, done in zip(notebook["cells"], ran.cells):
        if cell["cell_type"] != "code":
            continue
        outputs = []
        for output in done.outputs:
            data = output.get("data", {})
            if WIDGET in data:
                outputs += still(data[WIDGET]["model_id"], models)
            elif "image/png" in data:
                outputs.append(picture(base64.b64decode(data["image/png"])))
            else:
                outputs.append(output)
        cell["outputs"] = outputs
        cell["execution_count"] = done.execution_count
    lesson.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(lesson.name, sum(len(c.get("outputs", [])) for c in notebook["cells"]), "outputs,",
          f"{lesson.stat().st_size / 1e6:.1f} MB", flush=True)


def main():
    wanted = {f"{int(n):02d}" for n in sys.argv[1:]}
    lessons = sorted(p for p in COURSE.glob("lesson_*.ipynb")
                     if not p.stem.endswith("_solution") and (not wanted or p.name[7:9] in wanted))
    for lesson in lessons:
        fill(lesson)


if __name__ == "__main__":
    main()
