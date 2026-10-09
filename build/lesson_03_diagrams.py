"""Draws the lesson 3 diagrams into imgs/3 of the lecture folder, and the test photo.

coffee.png is scikit-image's coffee photo (CC0) enlarged twice with Lanczos,
so the canvas is big enough for the timing cells to tell the kernels apart.
The arithmetic is the notebook's own where it can be: the colour conversions,
gaussian_1d and the solved blur come from the NumPy cells of
lesson_03_notebook.py.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "imgs" / "3"
sys.path.insert(0, str(HERE))
import lesson_03_notebook as nb  # noqa: E402

ORANGE = "#F58518"
BLUE = "#4C78A8"
GREY = "#6B6B6B"
LIGHT = "#DDE6F0"

space = {"np": np}
for source in (nb.HELPERS_NP, nb.SOLUTION_BLUR_NP):
    exec(source, space)
srgb_to_linear, linear_to_srgb, lightness = space["srgb_to_linear"], space["linear_to_srgb"], space["lightness"]
gaussian_1d, blur = space["gaussian_1d"], space["blur"]

KERNELS = {
    "průměr": np.full((3, 3), 1 / 9),
    "zaostření": np.array(nb.SHARPEN, float).reshape(3, 3),
    "hrany": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], float),
    "reliéf": np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], float),
}


def make_photo():
    """coffee.png: scikit-image's coffee, 600 x 400, enlarged to the lesson's canvas."""
    import skimage.data
    small = Image.fromarray(skimage.data.coffee())
    big = small.resize((nb.WIDTH, nb.HEIGHT), Image.LANCZOS)
    big.save(OUT / "coffee.png", optimize=True)


def load(name):
    """A picture of imgs/3 as (h, w, 4) floats from 0 to 1, as Sara's import reads it."""
    rgb = np.asarray(Image.open(OUT / name).convert("RGB"), np.float32) / 255
    return np.dstack([rgb, np.ones(rgb.shape[:2], np.float32)])


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def bare(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def convolve3(img, w):
    """The 3 x 3 convolution of the notebook's check, edges clamped, colour only."""
    h, wd = img.shape[:2]
    p = np.pad(img[..., :3], ((1, 1), (1, 1), (0, 0)), mode="edge")
    return sum(w[j, i] * p[j:j + h, i:i + wd] for j in range(3) for i in range(3))


def numbers(ax, values, fmt, colours=None):
    """A grid of values with each number written in its cell."""
    n_rows, n_cols = values.shape
    for j in range(n_rows):
        for i in range(n_cols):
            colour = colours[j][i] if colours else ("white" if values[j, i] < 0.5 else "black")
            ax.text(i, j, fmt(values[j, i]), ha="center", va="center", fontsize=9, color=colour)


def convolution_window():
    """A 7 x 7 patch of grey values, the 3 x 3 window, its weights and the pixel it writes."""
    L = lightness(load("coffee.png"))
    patch = np.round(L[590:597, 653:660], 2)   # a clean edge found by eye, rounded as the labels print it
    w = KERNELS["zaostření"]
    out = np.zeros((7, 7))
    pad = np.pad(patch, 1, mode="edge")
    for j in range(7):
        for i in range(7):
            out[j, i] = (w * pad[j:j + 3, i:i + 3]).sum()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9), gridspec_kw={"width_ratios": [7, 3, 7], "wspace": 0.35})
    ax = axes[0]
    ax.imshow(patch, cmap="gray", vmin=0, vmax=1)
    numbers(ax, patch, lambda v: f"{v:.2f}")
    ax.add_patch(Rectangle((1.5, 1.5), 3, 3, fill=False, ec=ORANGE, lw=3))
    ax.set_title("okolí pixelu $f$", fontsize=12)
    ax = axes[1]
    ax.imshow(w, cmap="RdBu_r", vmin=-5, vmax=5)
    numbers(ax, w, lambda v: f"{v:g}", [["white" if abs(v) > 3 else "black" for v in row] for row in w])
    ax.add_patch(Rectangle((-0.5, -0.5), 3, 3, fill=False, ec=ORANGE, lw=3))
    ax.set_title("váhy $w$", fontsize=12)
    ax = axes[2]
    ax.imshow(np.clip(out, 0, 1), cmap="gray", vmin=0, vmax=1)
    numbers(ax, np.clip(out, 0, 1), lambda v: f"{v:.2f}")
    ax.add_patch(Rectangle((2.5, 2.5), 1, 1, fill=False, ec=ORANGE, lw=3))
    ax.set_title("výsledek $g$", fontsize=12)
    for ax in axes:
        bare(ax)
    axes[1].text(-0.45, 0.5, "×", fontsize=26, ha="center", va="center", transform=axes[1].transAxes)
    axes[1].text(1.45, 0.5, "=", fontsize=26, ha="center", va="center", transform=axes[1].transAxes)
    fig.text(0.5, 0.0, "devět součinů sečtených do jednoho pixelu, pro každý pixel znovu",
             ha="center", fontsize=11, color=GREY)
    save(fig, "convolution_window.png")


def kernel_gallery():
    """The photo's spoon after each kernel of the notebook's table."""
    img = load("coffee.png")[300:620, 520:840]
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.2), gridspec_kw={"wspace": 0.05})
    axes[0].imshow(img[..., :3])
    axes[0].set_title("původní", fontsize=12)
    for ax, (name, w) in zip(axes[1:], KERNELS.items()):
        ax.imshow(np.clip(convolve3(img, w), 0, 1))
        ax.set_title(name, fontsize=12)
    for ax in axes:
        bare(ax)
    save(fig, "kernel_gallery.png")


def separable():
    """The 2D Gaussian as a column times a row, and how many neighbours each way reads."""
    sigma = 3.0
    k = gaussian_1d(sigma)
    fig = plt.figure(figsize=(12, 4.2))
    grid = fig.add_gridspec(2, 4, width_ratios=[1, 6, 1.5, 7], height_ratios=[1, 6], wspace=0.08, hspace=0.08)
    top = fig.add_subplot(grid[0, 1])
    top.bar(np.arange(len(k)), k, color=ORANGE, width=0.8)
    top.set_xlim(-0.5, len(k) - 0.5)
    bare(top)
    left = fig.add_subplot(grid[1, 0])
    left.barh(np.arange(len(k)), k, color=BLUE, height=0.8)
    left.set_ylim(len(k) - 0.5, -0.5)
    left.invert_xaxis()
    bare(left)
    square = fig.add_subplot(grid[1, 1])
    square.imshow(np.outer(k, k), cmap="magma", aspect="auto")
    bare(square)
    square.text(len(k) / 2 - 0.5, len(k) + 1.2, "sloupec × řádek = jádro 2D", ha="center", va="top", fontsize=11)
    plot = fig.add_subplot(grid[:, 3])
    r = np.arange(1, 25)
    plot.plot(r, (2 * r + 1) ** 2, color=GREY, lw=2.2, label="jeden průchod, $(2r + 1)^2$")
    plot.plot(r, 2 * (2 * r + 1), color=ORANGE, lw=2.2, label="dva průchody, $2(2r + 1)$")
    plot.axvline(9, color="#bbbbbb", lw=1, ls="--")
    plot.text(9.4, 1800, "$\\sigma = 3$", color=GREY, fontsize=10)
    plot.axvline(24, color="#bbbbbb", lw=1, ls="--")
    plot.text(23.6, 1800, "$\\sigma = 8$", color=GREY, fontsize=10, ha="right")
    plot.set_xlabel("poloměr $r$ (pixely)", fontsize=11)
    plot.set_ylabel("sousedů na pixel", fontsize=11)
    plot.legend(frameon=False, fontsize=10, loc="upper left")
    for s in ("top", "right"):
        plot.spines[s].set_visible(False)
    save(fig, "separable.png")


def edge_modes():
    """The photo shifted right and down, with each rule filling what came from beyond the edge."""
    img = load("coffee.png")[::4, ::4, :3]
    h, w = img.shape[:2]
    dx, dy = nb.SHIFT[0] // 4, nb.SHIFT[1] // 4
    modes = [("Zero", "constant"), ("Clamp", "edge"), ("Mirror", "symmetric"), ("Wrap", "wrap")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 2.75), gridspec_kw={"wspace": 0.06})
    for ax, (name, mode) in zip(axes, modes):
        p = np.pad(img, ((dy, 0), (dx, 0), (0, 0)), mode=mode)
        ax.imshow(p[:h, :w])
        ax.axvline(dx - 0.5, color=ORANGE, lw=1.6, ls="--")
        ax.axhline(dy - 0.5, color=ORANGE, lw=1.6, ls="--")
        ax.set_title(name, fontsize=12)
        bare(ax)
    fig.text(0.5, -0.02, "vlevo a nahoře od čárkovaných čar je to, co kernel přečetl za okrajem fotky",
             ha="center", fontsize=11, color=GREY)
    save(fig, "edge_modes.png")


def cells(ax, x0, y, count, colours, size=1.0, ec="white"):
    for i in range(count):
        ax.add_patch(Rectangle((x0 + i * size, y), size, size, fc=colours(i), ec=ec, lw=0.5))


def shared_tile():
    """One workgroup's row in shared memory: who loads which pixel, and what one thread sums."""
    group, r_max = 64, 24
    total = group + 2 * r_max
    top, threads, memory = 10.0, 5.0, 0.0
    fig, ax = plt.subplots(figsize=(13, 4.6))
    # the image row the group needs
    cells(ax, 0, top, total, lambda i: ORANGE if r_max <= i < r_max + group else LIGHT)
    ax.text(-2, top + 0.5, "řádek obrázku", ha="right", va="center", fontsize=11)
    ax.text(r_max / 2, top + 1.5, f"{r_max} vlevo", ha="center", fontsize=10, color=GREY)
    ax.text(r_max + group / 2, top + 1.5, f"{group} pixelů skupiny", ha="center", fontsize=10, color=ORANGE)
    ax.text(total - r_max / 2, top + 1.5, f"{r_max} vpravo", ha="center", fontsize=10, color=GREY)
    # the threads and what each loads: thread t loads pixel t and, for t < 48, pixel t + 64
    cells(ax, r_max, threads, group, lambda i: BLUE)
    ax.text(-2, threads + 0.5, "vlákna skupiny", ha="right", va="center", fontsize=11)
    for t in (0, 20, 47):
        for target in (t, t + group):
            ax.add_patch(FancyArrowPatch((r_max + t + 0.5, threads + 1), (target + 0.5, top), arrowstyle="-|>",
                                         mutation_scale=8, color=BLUE, lw=0.9, alpha=0.8))
    ax.text(total + 2, (top + threads + 1) / 2, "1. načíst: vlákno $t$ přečte\npixely $t$ a $t + 64$",
            fontsize=10, va="center")
    ax.text(total + 2, threads + 0.5 - 2, "2. barrier(): počkat\nna celou skupinu", fontsize=10, va="center")
    # one thread's window
    t = 30
    centre = t + r_max
    cells(ax, 0, memory, total, lambda i: ORANGE if abs(i - centre) <= r_max else "#f3f3f3")
    ax.add_patch(Rectangle((centre, memory), 1, 1, fc="black", ec="white", lw=0.5))
    ax.add_patch(FancyArrowPatch((centre + 0.5, threads), (centre + 0.5, memory + 1), arrowstyle="<|-",
                                 mutation_scale=10, color="black", lw=1.2))
    ax.text(-2, memory + 0.5, "sdílená paměť", ha="right", va="center", fontsize=11)
    ax.text(total + 2, memory + 0.5, f"3. vlákno {t} sečte {2 * r_max + 1}\nhodnot ze sdílené paměti",
            fontsize=10, va="center")
    ax.set_xlim(-16, total + 20)
    ax.set_ylim(memory - 0.8, top + 2.4)
    bare(ax)
    save(fig, "shared_tile.png")


def oklab_to_linear(lab):
    l_ = lab[..., 0] + 0.3963377774 * lab[..., 1] + 0.2158037573 * lab[..., 2]
    m_ = lab[..., 0] - 0.1055613458 * lab[..., 1] - 0.0638541728 * lab[..., 2]
    s_ = lab[..., 0] - 0.0894841775 * lab[..., 1] - 1.2914855480 * lab[..., 2]
    lms = np.stack([l_ ** 3, m_ ** 3, s_ ** 3], axis=-1)
    return lms @ np.array([[4.0767416621, -3.3077115913, 0.2309699292],
                           [-1.2684380046, 2.6097574011, -0.3413193965],
                           [-0.0041960863, -0.7034186147, 1.7076147010]]).T


def shade(g):
    """The kernel's shade() with gain 1 and direction on, for an (..., 2) array of gradients."""
    m = np.clip(np.hypot(g[..., 0], g[..., 1]), 0, 1)
    unit = g / np.maximum(np.hypot(g[..., 0], g[..., 1]), 1e-9)[..., None]
    lab = np.dstack([0.85 * m, 0.15 * m * unit[..., 0], 0.15 * m * unit[..., 1]])
    return np.clip(linear_to_srgb(np.maximum(oklab_to_linear(lab), 0)), 0, 1)


def sobel():
    """The two kernels, gradients on a patch of the photo, and the colour each direction gets."""
    gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), gridspec_kw={"width_ratios": [3, 3, 5, 4], "wspace": 0.25})
    for ax, w, name in ((axes[0], gx, "$G_x$"), (axes[1], gx.T, "$G_y$")):
        ax.imshow(w, cmap="RdBu_r", vmin=-3, vmax=3)
        numbers(ax, w, lambda v: f"{v:g}", [["black"] * 3] * 3)
        ax.set_title(name, fontsize=13)
        bare(ax)
    L = lightness(load("coffee.png"))[380:460, 600:700]
    p = np.pad(L, 1, mode="edge")
    h, w = L.shape
    g = np.zeros((h, w, 2))
    for j in range(3):
        for i in range(3):
            g[..., 0] += gx[j, i] * p[j:j + h, i:i + w]
            g[..., 1] += gx.T[j, i] * p[j:j + h, i:i + w]
    ax = axes[2]
    ax.imshow(L, cmap="gray", vmin=0, vmax=1)
    step = 6
    ys, xs = np.mgrid[step // 2:h:step, step // 2:w:step]
    gs = g[ys, xs]
    strong = np.hypot(gs[..., 0], gs[..., 1]) > 0.15
    ax.quiver(xs[strong], ys[strong], gs[..., 0][strong], gs[..., 1][strong], color=ORANGE,
              angles="xy", scale_units="xy", scale=0.12, width=0.006)
    ax.set_title("gradient míří od tmavé ke světlé", fontsize=12)
    bare(ax)
    ax = axes[3]
    s = np.linspace(-1, 1, 241)
    y, x = np.meshgrid(s, s, indexing="ij")
    wheel = shade(np.dstack([x, y]))
    wheel[np.hypot(x, y) > 1] = 1
    ax.imshow(wheel, extent=(-1, 1, 1, -1))
    ax.text(1.06, 0, "světlejší\nvpravo", ha="left", va="center", fontsize=9, color=GREY)
    ax.text(0, 1.06, "světlejší dole", ha="center", va="top", fontsize=9, color=GREY)
    ax.set_title("barva podle směru", fontsize=12)
    bare(ax)
    save(fig, "sobel.png")


def numpy_shifts():
    """A padded row and the five slices a blur of radius 2 sums."""
    k = gaussian_1d(2 / 3)
    r = len(k) // 2
    w = 8
    values = np.array([0.2, 0.25, 0.3, 0.8, 0.85, 0.8, 0.3, 0.2])
    padded = np.pad(values, r, mode="edge")
    fig, ax = plt.subplots(figsize=(9, 4.6))
    cmap = plt.get_cmap("gray")
    cells(ax, 0, 6, len(padded), lambda i: cmap(padded[i]), ec=GREY)
    for i in (0, 1, len(padded) - 2, len(padded) - 1):
        ax.add_patch(Rectangle((i, 6), 1, 1, fill=False, ec=ORANGE, lw=1.2, ls="--"))
    ax.text(-0.6, 6.5, "p", ha="right", va="center", fontsize=12, family="monospace")
    for i in range(len(k)):
        y = 4.5 - i * 1.1
        cells(ax, i, y, w, lambda c: cmap(padded[i + c]), ec=GREY)
        ax.text(-0.6, y + 0.5, f"{k[i]:.3f} ×", ha="right", va="center", fontsize=10)
        ax.text(i + w + 0.4, y + 0.5, f"p[{i}:{i} + w]", va="center", fontsize=10, family="monospace")
    ax.text(len(padded) / 2, 7.4, "řádek rozšířený o r = 2 pixely z každé strany (okraj natažený)",
            ha="center", fontsize=10, color=GREY)
    ax.set_xlim(-3, len(padded) + 4)
    ax.set_ylim(-1.2, 7.9)
    ax.set_aspect("equal")
    bare(ax)
    save(fig, "numpy_shifts.png")


def second_derivative():
    """A soft edge sampled at pixels, its first and second differences, and the edge minus the second."""
    x = np.arange(24)
    f = 0.2 + 0.6 / (1 + np.exp(-(x - 11.5) / 0.9))
    p = np.pad(f, 1, mode="edge")
    first = (p[2:] - p[:-2]) / 2
    second = p[2:] - 2 * p[1:-1] + p[:-2]
    panels = [(f, "světlost $f$", GREY), (first, "první derivace, $(f_{+1} - f_{-1}) / 2$", BLUE),
              (second, r"druhá derivace, jádro $(1\;\,{-2}\;\,1)$", ORANGE),
              (f - second, r"$f - f''$, zaostření", GREY)]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.0), gridspec_kw={"wspace": 0.28})
    for ax, (values, title, colour) in zip(axes, panels):
        ax.axvline(11.5, color="#cccccc", lw=1, ls="--", zorder=0)
        ax.axhline(0, color="#cccccc", lw=1, zorder=0)
        ax.plot(x, values, color=colour, lw=1.4)
        ax.plot(x, values, "o", color=colour, ms=3.5)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[3].plot(x, f, color=GREY, lw=1, ls=":", zorder=0)
    axes[2].annotate("průchod nulou\nna hraně", (11.5, 0), (14.5, 0.03), fontsize=10, color=GREY,
                     arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8))
    save(fig, "second_derivative.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    make_photo()
    convolution_window()
    kernel_gallery()
    separable()
    edge_modes()
    shared_tile()
    sobel()
    second_derivative()
    numpy_shifts()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
