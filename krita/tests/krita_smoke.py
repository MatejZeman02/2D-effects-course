"""The template inside a headless Krita, through kritarunner.

kritarunner imports scripts from its own pykrita folder, so this file and the
pga_filter folder are linked there first (krita/tests/run_in_krita.sh does it).
The first argument is a folder holding NumPy for Krita's Python, the second
the lecture folder. Results go to smoke.log beside this file and to stdout of
the runner, one line a check, and the last line says how many failed.
"""

import os
import sys
import traceback

LOG = []


def check(name, ok, detail=""):
    LOG.append(f"{'ok  ' if ok else 'FAIL'} {name}" + (f"  ({detail})" if str(detail) else ""))
    return ok


def __main__(args):
    vendor, root = args[0], args[1]
    sys.path.insert(0, vendor)
    out = os.path.join(root, "krita", "tests", "smoke.log")
    try:
        run(root)
    except Exception:
        LOG.append("FAIL crashed\n" + traceback.format_exc())
    failed = sum(line.startswith("FAIL") for line in LOG)
    LOG.append(f"{failed} failed")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(LOG) + "\n")
    return 0


def run(root):
    import pga_filter  # registers the extension and puts its own NumPy on the path, as in Krita
    from pga_filter import dialog, engine, params, pixels

    import numpy as np
    from krita import Krita

    check("NumPy comes from " + ("the plugin" if "pga_filter" in np.__file__ else "outside"), True, np.__file__)

    app = Krita.instance()
    chart = os.path.join(root, "imgs", "0", "color_chart.png")
    effect = engine.load()
    check("effect.py loads with its title", effect.title == "Invert", effect.title)

    for depth in ("U8", "U16", "F16", "F32"):
        doc = app.openDocument(chart)
        if depth != "U8":
            doc.setColorSpace("RGBA", depth, "")
            doc.waitForDone()
        node = doc.rootNode().childNodes()[0]
        doc.setActiveNode(node)
        check(f"{depth} layer passes the checks", node.colorDepth() == depth and _ok(pixels.check_layer, node))
        red = pixels.read_layer(node, 200, 100, 1, 1)[0, 0]
        check(f"{depth} reads the red swatch as RGBA", np.allclose(red, [1, 0, 0, 1], atol=0.01), red.round(3))

        before = len(doc.rootNode().childNodes())
        name = node.name()
        copy = engine.apply_to_layer(doc, node, effect, {"amount": 1.0})
        doc.waitForDone()
        cyan = pixels.read_layer(copy, 200, 100, 1, 1)[0, 0]
        check(f"{depth} invert turns red cyan", np.allclose(cyan, [0, 1, 1, 1], atol=0.01), cyan.round(3))
        check(f"{depth} the result replaces the layer under its name",
              len(doc.rootNode().childNodes()) == before and copy.name() == name
              and (app.activeWindow() is None or doc.activeNode().name() == name),
              f"{len(doc.rootNode().childNodes())} layers")
        doc.close()

    # A selection over the left half changes only the left half.
    doc = app.openDocument(chart)
    node = doc.rootNode().childNodes()[0]
    w, h = doc.width(), doc.height()
    sel = doc.selection() or _selection(app)
    sel.select(0, 0, w // 2, h, 255)
    doc.setSelection(sel)
    check("the region is the selection's bounds", pixels.region(doc) == (0, 0, w // 2, h), pixels.region(doc))
    right_before = pixels.read_layer(node, w // 2, 0, w - w // 2, h)
    copy = engine.apply_to_layer(doc, node, effect, {"amount": 1.0})
    doc.waitForDone()
    right_after = pixels.read_layer(copy, w // 2, 0, w - w // 2, h)
    left = pixels.read_layer(copy, 200, 100, 1, 1)[0, 0]
    check("outside the selection nothing moves", np.array_equal(right_before, right_after))
    check("inside the selection the effect ran", np.allclose(left, [0, 1, 1, 1], atol=0.01), left.round(3))
    doc.close()

    # A group is refused in words.
    doc = app.createDocument(64, 64, "group", "RGBA", "U8", "", 72.0)
    group = doc.createNode("group", "grouplayer")
    doc.rootNode().addChildNode(group, None)
    try:
        pixels.check_layer(group)
        check("a group layer is refused", False)
    except pixels.PixelError as error:
        check("a group layer is refused", "paint layers only" in str(error), str(error)[:60])

    # apply() that fails reaches the dialog as text, not as a crash.
    broken = engine.Effect("Broken", params.parse({"k": (1.0, 0.0, 2.0)}), lambda img, k: img[..., :3])
    try:
        engine.run(broken, np.zeros((4, 4, 4), np.float32), {"k": 1.0})
        check("a wrong shape is reported", False)
    except engine.EffectError as error:
        check("a wrong shape is reported", "shape" in str(error))

    # The dialog with one control of every kind, drawn to a picture.
    every = {
        "amount": (0.5, 0.0, 1.0),
        "width": (16.0, 2.0, 64.0, 0.5),
        "steps": (4, 1, 8),
        "tint": "#ff8c32",
        "veil": "#3070ff80",
        "mode": ["Multiply", "Screen"],
        "invert": False,
        "weights": [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
    }
    seen = {}

    def show_all(img, amount, width, steps, tint, veil, mode, invert, weights):
        seen.update(amount=amount, width=width, steps=steps, tint=tint, veil=veil,
                    mode=mode, invert=invert, weights=weights)
        out = img.copy()
        out[..., :3] = out[..., :3] * (1 - amount) + np.asarray(tint, np.float32) * amount
        return out

    doc = app.openDocument(chart)
    node = doc.rootNode().childNodes()[0]
    original = pixels.read_layer(node, *pixels.region(doc))
    fx = engine.Effect("Every kind", params.parse(every), show_all)
    dlg = dialog.FilterDialog(fx, doc, node, original)
    check("the dialog hands apply() the right kinds",
          isinstance(seen.get("steps"), int) and isinstance(seen.get("amount"), float)
          and len(seen.get("tint", ())) == 3 and len(seen.get("veil", ())) == 4
          and seen.get("mode") == "Multiply" and seen.get("invert") is False
          and getattr(seen.get("weights"), "shape", None) == (3, 3),
          {k: type(v).__name__ for k, v in seen.items()})
    dlg.controls["steps"].box.setValue(7)
    dlg.controls["mode"].setCurrentIndex(1)
    dlg.redraw()
    check("moving controls reaches apply()", seen.get("steps") == 7 and seen.get("mode") == "Screen",
          f"steps {seen.get('steps')}, mode {seen.get('mode')}")
    dlg.adjustSize()
    picture = os.path.join(root, "krita", "tests", "dialog.png")
    check("the dialog draws", dlg.grab().save(picture), picture)
    doc.close()


def _ok(fn, *args):
    try:
        fn(*args)
        return True
    except Exception:
        return False


def _selection(app):
    from krita import Selection
    return Selection()
