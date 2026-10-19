# 2D effects course

Notebooks and a Krita plugin template for the 2D part of the computer graphics
course at FIT ČVUT, winter semester 2026/27. Each lesson builds one image
effect three ways: as a GPU kernel in Sára, rerun live from a Jupyter notebook
as you move its sliders, as a plain NumPy function, and as a Krita 6 plugin
running that same function.

The notebooks are in Czech. The code, its comments and the Krita plugin are in
English.

## What is here

| Path | What it holds |
|---|---|
| `lesson_NN_*.ipynb` | the lessons, one notebook each |
| `lib/` | the Python client for Sára (`sara.py`, `sara_notebook.py`, `sara_live.py`, `sara_params.py`) and the JupyterLab extension that highlights `%%gmacs` cells |
| `imgs/<n>/` | the pictures and test images of lesson `n` |
| `krita/` | the Krita plugin template `pga_filter`, its manual and its tests |
| `build/` | the scripts that generate the notebooks, their pictures and the plugin zips |
| `AGENTS.md` | instructions for AI coding assistants, which all of them read |

## Getting started

1. Get Sára, which comes as a separate zip for Windows or Linux, handed out in
   the course.
2. Install Python 3.13, the version Krita 6 runs, so an effect behaves the
   same in the notebook and in the plugin. `requirements.txt` pins the
   plugin's NumPy, 2.5.1, for the same reason. A computer often has an older
   Python as its default (the lab's is 3.10), so name the version when you
   create the virtual environment, `.venv`, in the repository's folder. On
   Windows, in PowerShell:

   ```powershell
   py --list
   py -3.13 -m venv .venv
   .venv\Scripts\Activate.ps1
   python --version
   pip install -r requirements.txt
   ```

   `py --list` shows the versions installed. If 3.13 is not among them,
   install it from [python.org](https://www.python.org/downloads/), which
   needs no administrator rights when installed for your user only. On Linux:

   ```bash
   python3.13 -m venv .venv
   source .venv/bin/activate
   python --version
   pip install -r requirements.txt
   ```

   `python --version` has to say 3.13. If it does not, the environment was
   made with another Python: delete the `.venv` folder and create it again.
   Keep the `-r`: without it `pip` reads `requirements.txt` as the name of a
   package. The prompt starts with `(.venv)` while the environment is active.
   If PowerShell refuses `Activate.ps1` because running scripts is disabled,
   run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use
   `cmd` and `.venv\Scripts\activate.bat` instead.

3. Start Sára with its agent door (`sara-with-door.cmd` on Windows,
   `./sara.x86_64 --display-driver wayland -- --agent-door` on Linux). On an
   X11 desktop leave out `--display-driver wayland`. Without it on Wayland,
   Sára runs under XWayland and has to guess its interface scale.
4. Open lesson 0, either in JupyterLab, started with `jupyter lab` while the
   environment is active, or in VS Code as below.

## The notebooks in VS Code

1. Install two extensions, both by Microsoft: **Python** and **Jupyter**.
   Jupyter brings **Jupyter Notebook Renderers** along, which draws the
   sliders and pictures. From a terminal:

   ```bash
   code --install-extension ms-python.python
   code --install-extension ms-toolsai.jupyter
   ```

2. Open the repository's folder with **File > Open Folder**, not a single
   notebook, so VS Code finds `.venv`.
3. Open a lesson, click **Select Kernel** at the top right, choose **Python
   Environments** and pick `.venv`, which VS Code lists with its Python
   version, 3.13. The kernel is the Python process that runs
   the notebook's cells, and `requirements.txt` already installed it
   (`ipykernel`, through JupyterLab). If VS Code offers to install
   `ipykernel`, another environment was picked: choose `.venv` again.
4. Run the first cell.

Every notebook comes with the outputs of its solved lesson, so each cell already
shows how it should come out. An interactive picture shows as a still picture
until its cell runs, and then its sliders appear. A task cell's output turns
into what your code makes once you run it, so look at it before and after.

If a cell runs but shows "Error rendering output item using
'jupyter-ipywidget-renderer'" and "Cannot read properties of undefined
(reading 'ipywidgetsKernel')" in place of its slider and picture, VS Code's
widget renderer failed to load. Press **Ctrl+Shift+P**, run **Developer:
Reload Window** and run the cell again. A dialog saying that widgets need
supporting files downloaded from a third party website (`@jupyter-widgets/base`)
is the same fault, and the reload fixes it too.

> [!NOTE]
> VS Code's Jupyter 2025.9.1 cannot draw an `ipywidgets.Output` and shows a
> red box, "Failed to load model class 'OutputModel' from module
> '@jupyter-widgets/output'", in its place. The lessons' cells use none, so
> their sliders and pictures show in VS Code as well. An `ipywidgets.Output` in
> your own code, or in a library's, can still show the red box there, and works
> in JupyterLab.

## Highlighting gmacs cells

A notebook cell whose first line is `%%gmacs` holds a Sára kernel. Two
extensions colour such a cell as gmacs, one for JupyterLab in `lib/` and one
for VS Code, and every other cell stays Python. There is no language to pick:
the cell switches as soon as you type `%%gmacs` on its first line, and
switches back when you delete it.

`pip install -r requirements.txt` already installs the extension. To add it to
a JupyterLab you have already set up, install the wheel with the `pip` of the
environment that runs JupyterLab, from the root of this repository:

```bash
pip install lib/jupyterlab_gmacs_syntax-0.1.2-py3-none-any.whl
```

Then stop JupyterLab, start it again and reload the browser tab. Check that it
is on with:

```bash
jupyter labextension list
```

which lists `jupyterlab-gmacs-syntax` as enabled. It works in JupyterLab 4 and
Jupyter Notebook 7, and needs no Node and no rebuild.

VS Code does not load JupyterLab extensions, so it has its own. Download
`gmacs-syntax-<version>.vsix` from its
[releases page](https://github.com/MatejZeman02/gmacs-acompute-syntax-highlighting-vs-code-extension/releases/latest),
install it once with the version you downloaded, then run **Developer: Reload
Window**:

```bash
code --install-extension gmacs-syntax-1.2.0.vsix
```

VS Code installs **Godot Tools** alongside it, whose shader grammar colours the
GLSL inside a kernel. The classic Notebook 6 colours gmacs cells as Python. The
colouring is cosmetic: the `%%gmacs` cells run either way.

## The Krita plugin

Download `pga_filter-windows.zip` or `pga_filter-linux.zip` from the
repository's releases and import it in Krita 6 through **Tools > Scripts >
Import Python Plugin from File**. Each zip carries the NumPy build that matches
Krita 6's Python 3.13 on that system. The effect lives in one file,
`pga_filter/effect.py`: a dict `PARAMS` and a function `apply(img, ...)`.
[`Manual.html`](krita/pga_filter/Manual.html) covers the rest.

To build the zips yourself:

```bash
python build/krita_template.py --download
```

The unit tests run in plain Python with `python -m pytest krita/tests`.

> [!NOTE]
> Every notebook is generated by `build/lesson_NN_notebook.py` and its pictures
> by `build/lesson_NN_diagrams.py`. Change the script and run it again, because
> the next build overwrites any edit made in the notebook itself. The build
> writes the notebooks without outputs. With Sára running,
> `python build/fill_outputs.py` then runs every solved lesson and copies its
> outputs into the notebooks, `python build/fill_outputs.py 4` just lesson 4.

## Licence

GPL-3.0, see [LICENSE](LICENSE).
