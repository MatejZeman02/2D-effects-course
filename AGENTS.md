# Instructions for AI assistants

This repository holds the notebooks and the Krita plugin template for the 2D
part of the computer graphics course at FIT ČVUT. Whoever opened it is most
likely a student working through the lessons. These instructions are for
whichever coding assistant they use. Claude Code reads `CLAUDE.md` and Gemini
CLI reads `GEMINI.md`, both of which import this file, and Codex, Cursor and
GitHub Copilot read this file directly.

## Help the student learn

- Every lesson sets tasks marked **🎯 Úkol**, and the point is that the student
  writes them. Explain the idea, name the section of the notebook that covers
  it and give a hint first. Write the finished code only when the student asks
  for it outright.
- Each task already has a collapsed **💡 Nápověda** and a separate
  **🔑 Řešení** in the notebook. Point to those before writing your own.
- Each notebook opens with a table of the functions it introduces, and lesson
  0 ends with a full overview of Sára's Python client.
- Reply in the language the student writes in. The notebooks are in Czech, the
  code and its comments in English.
- A **✅ Kontrola** cell compares the student's result with the expected one.
  A difference around 0.001 is the GPU rounding differently from the CPU, not a
  bug.

## Commits and attribution

- Never add a `Co-Authored-By` trailer, a "Generated with" line or any other AI
  attribution to a commit message, a pull request or a file.
- Students cannot push to this public repository, only to a fork or a new
  repository of their own, which may later hold their semester project. Before
  you push, check that the remote is no longer
  https://github.com/MatejZeman02/2D-effects-course.
- `.claude/settings.json` turns Claude Code's attribution off in this
  repository.

## What is where

| Path | What it is | Edit it? |
|---|---|---|
| `lesson_NN_*.ipynb` | the lessons | work in a copy, see the tip below |
| `lib/` | Sára's Python client and the JupyterLab highlighter, copied from Sára's releases | no, `git pull` replaces it |
| `krita/pga_filter/effect.py` | the effect the Krita plugin runs | yes, the only file of the plugin to edit |
| `krita/pga_filter/` (the rest) | the plugin template | no |
| `imgs/` | the pictures and test images | no |
| `build/` | the scripts that generate the notebooks, their pictures and the plugin zips | only for a change to the course itself |

> [!TIP]
> Suggest that the student works in a copy of each notebook, saved next to the
> original (for example `lesson_01_moje.ipynb`) so that `lib/` and `imgs/` are
> still found. Then `git pull` brings the course's updates without a merge
> conflict.

## Sára and the notebook

- Sára is a painting program that computes on the GPU. It comes as a zip handed
  out in the course, not from this repository.
- A notebook talks to a running Sára over a local connection, its agent door.
  Start Sára with `sara-with-door.cmd` on Windows or
  `./sara.x86_64 --display-driver wayland -- --agent-door` on Linux (without
  `--display-driver wayland` on an X11 desktop), or tick **Modules > Local
  Server** in a Sára that is already running.
- Sára serves one notebook kernel at a time. `Sara hung up with a step
  outstanding` means `sara.init()` ran a second time in one kernel, or another
  lesson's kernel is still alive: restart this kernel and shut the other down.
- `sara.init()` replaces the document open in Sára with a new sRGB one. Warn
  the student first if they may have unsaved work there.
- You cannot see Sára's window, but you can look at its picture through the
  door, as the next section shows.
- An image is a NumPy `float32` array of shape (height, width, 4), RGBA from 0
  to 1, sRGB encoded.
- Every function of the client has a docstring, `help(sara.live)` or
  `help(doc.new_layer)`. Read `lib/sara.py` rather than guess an API.
- In VS Code an `ipywidgets.Output` shows a red box, "Failed to load model
  class 'OutputModel'", instead of its content. That is VS Code's renderer,
  not the student's code. The lessons' cells use no `Output`, so avoid it in
  code you write for the student too, or suggest JupyterLab where one is
  needed.

## Using Sára's door yourself

The door is not only the notebook's. You can attach to the same Sára from a
shell, list its layers, read their pixels, save the picture as a PNG to look
at and check that a kernel compiles, all through the client in `lib/sara.py`.

- Sára must be running with its door open, as above. **Modules > Connect an
  agent** also copies a message for an agent, which the student may paste into
  the conversation. It describes the door's raw HTTP, but use `lib/sara.py`,
  which speaks it for you and finds the door by itself.
- Run Python from the repository's root with `lib` on the path. Attach with
  `sara.connect()`, which leaves the open document as it is, and never with
  `sara.init()`, which replaces it.
- **The door serves one client at a time.** While the notebook's kernel holds
  it, your connection is closed unanswered and `sara.connect()` raises a
  `SaraError`. Ask the student to run `doc.close()` in the notebook, or to
  restart its kernel, before you attach. Sára also lets go of a client that
  has been idle for ten minutes. Hang up as soon as you are done, which the
  `with` block does, and the notebook's next call attaches again by itself.
- Read freely, but ask before you write. A write, a new layer, an imported
  picture or a kernel run changes the student's picture, each as one undo step
  they can take back in Sára.

Reading the document and looking at it:

```python
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "lib")
import sara

with sara.connect() as doc:
    for entry in doc.layers():                    # the stack, bottom first
        print(entry["i"], entry["name"], entry["visible"])
    a = doc.layer("Background").read()            # float32 (h, w, 4), sRGB
    print(a.shape, a[..., :3].mean(axis=(0, 1)))
    look = Path(tempfile.gettempdir()) / "sara-look.png"
    look.write_bytes(doc.show().data)             # the whole picture, 512 px on the long side
```

Then open that PNG with your own image reading to see what the student sees.
`doc.layer("Dithered").show().data` is one layer instead of the whole picture.

Checking that a kernel compiles, which touches no layer:

```python
source = """uniform float levels: hint_range(2, 16) = 4.0

def pixel(ivec2 at) -> vec4:
    return floor(src(at) * levels + 0.5) / levels
"""
with sara.connect() as doc:
    said = doc.kernel({"source": source, "name": "pixels", "run": False})
    print(said["params"])                         # the uniforms Sára found
```

The source is the cell's body without its `%%gmacs` line. `name` is `pixels`
for the short pixel form, or the name on the `kernel` line of a whole kernel.
A kernel that does not compile raises a `SaraError` carrying the compiler's
message and the line it points at.

## gmacs kernels

A notebook cell whose first line is `%%gmacs "Source" -> "Target"` is not
Python. It is a GPU kernel in gmacs, Sára's dialect of GLSL written with
Python's block syntax. It reads the layer "Source" and writes the layer
"Target".

- Blocks go by indentation, four spaces, with no braces and no semicolons.
- GLSL's types, constructors, swizzles and built-in functions are unchanged:
  `vec4 c = src(at)`, `c.rgb`, `mix`, `clamp`, `floor`, `fract`, `dot`.
- A function is `def blend(vec3 a, vec3 b, float t) -> vec3:`. Control flow is
  `if`, `elif`, `else`, `while`, `and`, `or`, `not` and
  `for i in range(a, b):`.
- A constant is GLSL's without the semicolon,
  `const float A[16] = float[16](...)`.
- The lessons use the short pixel form: `uniform` lines, then
  `def pixel(ivec2 at) -> vec4:` returning the colour of one pixel. `src(at)`
  reads the source layer.
- Each `uniform` becomes a widget under the cell:
  - `uniform float amount: hint_range(0, 1) = 1.0` is a slider, and with
    `int` it moves in whole numbers
  - `uniform int i: hint_enum("A", "B") = 0` is a drop-down, and the kernel
    gets the index of the choice
  - `uniform vec3 c: source_color = vec3(1.0, 0.5, 0.2)` is a colour picker
    handing the kernel sRGB numbers
  - `uniform bool b = false` is a checkbox
- Sára's libraries are imported by name, one per line, right under the
  `%%gmacs` line and above the `uniform` lines. An import pastes the whole
  library into the kernel, so there is no `from ... import`, and a name the
  library defines cannot be declared again.

### Common gmacs imports

| Import | What it gives a kernel |
|---|---|
| `linear_srgb_color_space` | `srgb_to_linear_srgb(c)` decodes sRGB into linear light, `linear_srgb_to_srgb(c)` encodes it back, `gamma(c, g)` raises each channel to a power |
| `oklab_color_space` | `linear_srgb_to_oklab(c)` and `oklab_to_linear_srgb(c)`, between linear RGB and Oklab, whose `.x` is the perceived lightness |
| `noise` | `sa_hash21(p)` one number from 0 to 1 per point, `sa_hash22(p)` two of them, `sa_value_noise(p)` smooth noise, `sa_value_fbm(p)` three octaves of it, `sa_voronoi(p)` the distance to the nearest of scattered points, all from 0 to 1 |
| `erf` | `sa_erf(x)`, the Gauss error function GLSL lacks, for a Gaussian falloff |
| `positive_channels` | `sa_positive_channels(c)` pulls a colour with a negative channel towards grey until none is negative, before a `pow` or a product |

- Every `c` is a `vec3`. Oklab is reached through linear RGB, so sRGB to
  Oklab is `linear_srgb_to_oklab(srgb_to_linear_srgb(c.rgb))` and back is
  `linear_srgb_to_srgb(oklab_to_linear_srgb(lab))`.
- The noise functions take a `vec2` measured in cells. `sa_hash21(vec2(at))`
  is white noise, a new number every pixel, and `sa_value_noise(vec2(at) / 32.0)`
  changes smoothly over about 32 pixels.
- The libraries live in Sára's own repository, not in this one. Use only the
  functions this table and the lessons name, and never invent others.

## NumPy and the Krita plugin

- The Krita plugin runs the same NumPy function as the notebook. `PARAMS` and
  `apply` go into `krita/pga_filter/effect.py` unchanged, and `TITLE` names the
  filter.
- Krita 6 runs Python 3.13 with only the NumPy the plugin's zip carries.
  `effect.py` may import NumPy and the standard library and nothing else, so no
  SciPy, OpenCV or Pillow.
- `apply(img, **params)` gets a `float32` array (height, width, 4), RGBA from 0
  to 1, and returns one of the same shape. Write it with whole-array NumPy
  operations, because a Python loop over the pixels is far too slow.
- `PARAMS` is a dict whose order is the order of the widgets:

  | Value | Widget | `apply` gets |
  |---|---|---|
  | `(1.0, 0.0, 1.0)` | slider with default, minimum and maximum, a fourth number is the step | a float |
  | `(5, 2, 16)` | slider in whole numbers, when all three are int | an int |
  | `"#1b1035"`, `"#1b1035ff"` | colour picker, with alpha when there are eight digits | a tuple of sRGB floats from 0 to 1 |
  | `["Bayer", "IGN"]` | drop-down, the first is the default | the chosen label |
  | `False` | checkbox | a bool |
  | `[[0, 2], [3, 1]]` | square grid of numbers | an (n, n) `float32` array |

- The plugin's manual is `krita/pga_filter/Manual.html`. The zips to import in
  Krita are on the repository's releases page.

## Running things

```bash
pip install -r requirements.txt
jupyter lab
python -m pytest build krita/tests
```

The tests check the course's own build, not the student's work.
