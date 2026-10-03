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
  attribution to a commit message, a pull request or a file. The work is the
  student's.
- Commit and push only when the student asks.
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
  `./sara.x86_64 -- --agent-door` on Linux, or pick **Help > Connect an agent**
  in a Sára that is already running.
- `sara.init()` replaces the document open in Sára with a new sRGB one. Warn
  the student first if they may have unsaved work there.
- You cannot see Sára's canvas. Ask the student, or read what `layer.show()`
  and `doc.show()` display in the notebook.
- An image is a NumPy `float32` array of shape (height, width, 4), RGBA from 0
  to 1, sRGB encoded.
- Every function of the client has a docstring, `help(sara.live)` or
  `help(doc.new_layer)`. Read `lib/sara.py` rather than guess an API.

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
- Sára's libraries are imported by name, `import linear_srgb_color_space`,
  `import oklab_color_space` or `import noise`. Use only the library functions
  a lesson introduces and never invent others.

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
