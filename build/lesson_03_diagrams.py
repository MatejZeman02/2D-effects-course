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
srgb_to_linear, linear_to_srgb = space["srgb_to_linear"], space["linear_to_srgb"]
gaussian_1d, blur = space["gaussian_1d"], space["blur"]

SHARPEN = nb.preset_weights("Own 3 x 3")      # the own grid's opening values, mirrored
GALLERY = {"Disc 25 x 25": "kruh 25 × 25", "Motion 25 x 1": "pohyb 25 × 1", "Sharpen 9 x 9": "zaostření 9 × 9",
           "Edges 5 x 5": "hrany 5 × 5", "Emboss 5 x 5": "reliéf 5 × 5"}


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
    w = SHARPEN
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


def convolve(img, w):
    """The convolution of section 2, f(x - i, y - j) times w(i, j), edges clamped, colour only."""
    r = w.shape[0] // 2
    h, wd = img.shape[:2]
    p = np.pad(img[..., :3], ((r, r), (r, r), (0, 0)), mode="edge")
    out = np.zeros((h, wd, 3))
    for j in range(-r, r + 1):
        for i in range(-r, r + 1):
            if w[j + r, i + r] != 0:
                out += w[j + r, i + r] * p[r - j:r - j + h, r - i:r - i + wd]
    return out


def kernel_gallery():
    """The photo's spoon after the presets of section 2."""
    img = load("coffee.png")[300:620, 520:840]
    fig, axes = plt.subplots(1, 1 + len(GALLERY), figsize=(16, 3.0), gridspec_kw={"wspace": 0.05})
    axes[0].imshow(img[..., :3])
    axes[0].set_title("původní", fontsize=12)
    for ax, (name, title) in zip(axes[1:], GALLERY.items()):
        ax.imshow(np.clip(convolve(img, nb.preset_weights(name)), 0, 1))
        ax.set_title(title, fontsize=12)
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


def to_oklab(rgb):
    """OKLab of linear RGB, (..., 3)."""
    lms = np.cbrt(rgb @ np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                                  [0.2119034982, 0.6806995451, 0.1073969566],
                                  [0.0883024619, 0.2817188376, 0.6299787005]]).T)
    return lms @ np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                           [1.9779984951, -2.4285922050, 0.4505937099],
                           [0.0259040371, 0.7827717662, -0.8086757660]]).T


def lightness(img):
    """OKLab L of an (h, w, 4) picture in sRGB, an (h, w) array."""
    return to_oklab(srgb_to_linear(img[..., :3]))[..., 0]


BILATERAL_AT = (600, 421)     # a pixel of the cup's red side, just below the edge to its white rim


def bilateral():
    """A patch across an edge and its centre's bilateral weights: by distance, by colour, and both."""
    sigma, sigma_color = 3.0, 0.05
    r = int(np.ceil(3 * sigma))
    x, y = BILATERAL_AT
    patch = load("coffee.png")[y - r:y + r + 1, x - r:x + r + 1, :3]
    lab = to_oklab(srgb_to_linear(patch.astype(np.float64)))
    jj, ii = np.mgrid[-r:r + 1, -r:r + 1]
    spatial = np.exp(-(ii ** 2 + jj ** 2) / (2 * sigma ** 2))
    colour = np.exp(-((lab - lab[r, r]) ** 2).sum(axis=-1) / (2 * sigma_color ** 2))
    panels = [(patch, "okolí pixelu"), (spatial, "váha podle vzdálenosti"),
              (colour, "váha podle barvy"), (spatial * colour, "bilaterální váha, součin")]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
    for ax, (image, title) in zip(axes, panels):
        ax.imshow(image, cmap=None if image.ndim == 3 else "magma", vmin=0, vmax=1, interpolation="nearest")
        ax.add_patch(Rectangle((r - 0.5, r - 0.5), 1, 1, fill=False, ec="#00e0ff", lw=1.6))
        ax.set_title(title, fontsize=11)
        bare(ax)
    save(fig, "bilateral.png")


# --- The brainstorm's ideas, each a few panels of the photo's cup and spoon. ----------
CROP = (slice(330, 570), slice(560, 800))     # the spoon and the cup's rim, 240 x 240


def crop():
    return load("coffee.png")[CROP][..., :3]


def idea(name, panels, width=3.0):
    """One row of (title, picture) panels, a grey picture drawn as lightness."""
    fig, axes = plt.subplots(1, len(panels), figsize=(width * len(panels), 2.45), gridspec_kw={"wspace": 0.05})
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(np.clip(img, 0, 1), cmap="gray" if img.ndim == 2 else None, vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title, fontsize=11)
        bare(ax)
    save(fig, name)


def box_mean(img, radius):
    """The mean of the (2 radius + 1) square around each pixel, edges clamped."""
    side = 2 * radius + 1
    h, w = img.shape[:2]
    p = np.pad(img, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
    c = np.cumsum(np.cumsum(np.pad(p, ((1, 0), (1, 0), (0, 0))), axis=0), axis=1)
    return (c[side:side + h, side:side + w] - c[:h, side:side + w] - c[side:side + h, :w] + c[:h, :w]) / side ** 2


def ideas():
    img = crop()
    whole = np.zeros_like(img) + load("coffee.png")[..., :3].mean(axis=(0, 1))
    idea("idea_mean_all.png", [("původní", img), ("průměr", whole)])
    side = 17
    h, w = img.shape[:2]
    cut = img[:h // side * side, :w // side * side]
    blocks = cut.reshape(h // side, side, w // side, side, 3).mean(axis=(1, 3)).repeat(side, 0).repeat(side, 1)
    idea("idea_blocks.png", [("původní", cut), ("průměr po blocích", blocks)])
    rng = np.random.default_rng(3)
    ys, xs = np.mgrid[0:h, 0:w]
    ys = np.clip(ys + rng.integers(-8, 9, (h, w)), 0, h - 1)
    xs = np.clip(xs + rng.integers(-8, 9, (h, w)), 0, w - 1)
    idea("idea_random.png", [("původní", img), ("náhodný soused do 8 pixelů", img[ys, xs])])
    small = Image.fromarray((img * 255).astype(np.uint8)).resize((w // 8, h // 8), Image.BOX)
    back = np.asarray(small.resize((w, h), Image.BILINEAR), np.float32) / 255
    idea("idea_resize.png", [("původní", img), (f"zmenšená 8krát ({w // 8} px)", np.asarray(small, np.float32) / 255),
                             ("zase zvětšená", back)])
    L = lightness(np.dstack([img, np.ones(img.shape[:2], np.float32)]))
    p = np.pad(L, 1, mode="edge")
    right = np.abs(p[1:-1, 2:] - p[1:-1, :-2])
    down = np.abs(p[2:, 1:-1] - p[:-2, 1:-1])
    idea("idea_diff.png", [("světlost", L), ("rozdíl vlevo a vpravo", 3 * right), ("rozdíl nahoře a dole", 3 * down)])
    idea("idea_threshold.png", [("světlost", L), ("světlost nad 0.6", (L > 0.6).astype(float))])
    soft = blur(srgb_to_linear(img.astype(np.float64)), 3.0)
    Ls = to_oklab(soft)[..., 0]
    idea("idea_dog.png", [("světlost", L), ("rozmazaná, $\\sigma = 3$", Ls), ("rozdíl, zesílený 4krát", 4 * np.abs(L - Ls))])


# --- Lens, box and Gauss on the spoon's highlights --------------------------------
SPOON = (slice(420, 720), slice(560, 900))    # the spoon and the saucer's rim, 300 x 340
LENS_R = 12                                    # the lens's disc and the box's radius


def fft_convolve(a, k):
    """a, (h, w, 3), convolved with the small kernel k, edges clamped, through the FFT."""
    h, w = a.shape[:2]
    kh, kw = k.shape
    pad = np.pad(a, ((kh, kh), (kw, kw), (0, 0)), mode="edge")
    big = np.zeros(pad.shape[:2])
    big[:kh, :kw] = k
    big = np.roll(big, (-(kh // 2), -(kw // 2)), (0, 1))
    out = np.fft.irfft2(np.fft.rfft2(pad, axes=(0, 1)) * np.fft.rfft2(big)[..., None], s=pad.shape[:2], axes=(0, 1))
    return out[kh:kh + h, kw:kw + w]


def highlights(img):
    """The photo in linear light with its clipped highlights eleven times brighter, as the scene was."""
    near_white = np.clip((img.max(axis=-1, keepdims=True) - 0.92) / 0.08, 0, 1)
    return srgb_to_linear(img) * (1 + 10 * near_white)


def blurred(kind):
    """The spoon blurred by a disc (the lens), a square or a Gaussian of about the same size."""
    img = load("coffee.png")[..., :3].astype(np.float64)
    j, i = np.mgrid[-LENS_R:LENS_R + 1, -LENS_R:LENS_R + 1]
    k = {"lens": (i * i + j * j <= LENS_R * LENS_R).astype(float),
         "box": np.ones(i.shape),
         "gauss": np.exp(-(i * i + j * j) / (2 * (LENS_R / 2.5) ** 2))}[kind]
    out = fft_convolve(highlights(img), k / k.sum())
    return linear_to_srgb(np.clip(out, 0, 1))[SPOON]


def lens_pictures():
    original = load("coffee.png")[..., :3][SPOON]
    lens, box, gauss = blurred("lens"), blurred("box"), blurred("gauss")
    idea("lens_blur.png", [("původní", original), ("objektiv (lens blur)", lens)], width=3.4)
    idea("box_vs_lens.png", [(f"průměr čtverce {2 * LENS_R + 1} × {2 * LENS_R + 1}", box),
                             ("objektiv", lens)], width=3.4)
    idea("blur_shapes.png", [("průměr čtverce", box), ("Gauss", gauss), ("objektiv", lens)], width=3.4)


def box_window():
    """A 9 x 9 patch with a 45 degree edge, the 3 x 3 window of equal weights and the box mean it gives."""
    n = 9
    j, i = np.mgrid[0:n, 0:n]
    patch = np.where(i + j > n - 1, 0.9, np.where(i + j < n - 1, 0.1, 0.5))
    pad = np.pad(patch, 1, mode="edge")
    out = np.array([[pad[y:y + 3, x:x + 3].mean() for x in range(n)] for y in range(n)])
    at = (4, 3)                                       # the pixel whose window is drawn, row and column
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3), gridspec_kw={"width_ratios": [9, 3, 9], "wspace": 0.3})
    ax = axes[0]
    ax.imshow(patch, cmap="gray", vmin=0, vmax=1)
    numbers(ax, patch, lambda v: f"{v:.1f}")
    ax.add_patch(Rectangle((at[1] - 1.5, at[0] - 1.5), 3, 3, fill=False, ec=ORANGE, lw=3))
    ax.set_title("okolí pixelu $f$, šikmá hrana", fontsize=12)
    ax = axes[1]
    ax.imshow(np.full((3, 3), 0.75), cmap="Blues", vmin=0, vmax=1)
    for y in range(3):
        for x in range(3):
            ax.text(x, y, "1/9", ha="center", va="center", fontsize=10)
    ax.add_patch(Rectangle((-0.5, -0.5), 3, 3, fill=False, ec=ORANGE, lw=3))
    ax.set_title("váhy, všechny stejné", fontsize=12)
    ax = axes[2]
    ax.imshow(out, cmap="gray", vmin=0, vmax=1)
    numbers(ax, out, lambda v: f"{v:.2f}")
    ax.add_patch(Rectangle((at[1] - 0.5, at[0] - 0.5), 1, 1, fill=False, ec=ORANGE, lw=3))
    ax.set_title("výsledek $g$, průměr okna", fontsize=12)
    for ax in axes:
        bare(ax)
    axes[1].text(-0.45, 0.5, "×", fontsize=26, ha="center", va="center", transform=axes[1].transAxes)
    axes[1].text(1.45, 0.5, "=", fontsize=26, ha="center", va="center", transform=axes[1].transAxes)
    fig.text(0.5, 0.02, "okno se posune na každý pixel, přes šikmou hranu vyjdou čtyři mezistupně, přes rovnou by vyšly dva",
             ha="center", fontsize=11, color=GREY)
    save(fig, "box_window.png")


# --- The brainstorm's new ideas --------------------------------------------------------
def shifted_copies(img, step, reach):
    """The mean of img's copies shifted by every (dx, dy) on a grid of step out to reach, edges clamped."""
    h, w = img.shape[:2]
    p = np.pad(img, ((reach, reach), (reach, reach), (0, 0)), mode="edge")
    shifts = range(-reach, reach + 1, step)
    return np.mean([p[reach + dy:reach + dy + h, reach + dx:reach + dx + w] for dy in shifts for dx in shifts], axis=0)


def noisy(img):
    """img with the goal's grain: each channel moved by up to 0.15 either way."""
    rng = np.random.default_rng(5)
    return np.clip(img + 0.3 * (rng.random(img.shape) - 0.5), 0, 1)


def oklab_of(img):
    return to_oklab(srgb_to_linear(img.astype(np.float64)))


def bilateral_np(img, sigma, sigma_color):
    """Section 6's filter in NumPy, the colour weight from OKLab, the mean in linear light."""
    r = int(np.ceil(3 * sigma))
    h, w = img.shape[:2]
    lab = np.pad(oklab_of(img), ((r, r), (r, r), (0, 0)), mode="edge")
    lin = np.pad(srgb_to_linear(img.astype(np.float64)), ((r, r), (r, r), (0, 0)), mode="edge")
    centre = lab[r:r + h, r:r + w]
    total, weight = np.zeros((h, w, 3)), np.zeros((h, w, 1))
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            d = lab[r + dy:r + dy + h, r + dx:r + dx + w] - centre
            k = np.exp(-(dx * dx + dy * dy) / (2 * sigma ** 2)) * np.exp(-(d * d).sum(-1, keepdims=True) / (2 * sigma_color ** 2))
            total += k * lin[r + dy:r + dy + h, r + dx:r + dx + w]
            weight += k
    return linear_to_srgb(total / weight)


def sobel_strength(L):
    p = np.pad(L, 1, mode="edge")
    h, w = L.shape
    gx = sum(i * (2 - abs(j)) * p[1 + j:1 + j + h, 1 + i:1 + i + w] for j in (-1, 0, 1) for i in (-1, 0, 1))
    gy = sum(j * (2 - abs(i)) * p[1 + j:1 + j + h, 1 + i:1 + i + w] for j in (-1, 0, 1) for i in (-1, 0, 1))
    return np.hypot(gx, gy)


def more_ideas():
    img = crop()
    h, w = img.shape[:2]
    idea("idea_shifted.png", [("původní", img), ("9 kopií po 12 pixelech", shifted_copies(img, 12, 12)),
                              ("kopie po 1 pixelu do 8", shifted_copies(img, 1, 8))])
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.75), gridspec_kw={"wspace": 0.08})
    y, x = 62, 150                                    # pixel A on the edge of the cup's white rim, B is its right neighbour
    zoom = img[y - 9:y + 10, x - 9:x + 11]
    axes[0].imshow(zoom, interpolation="nearest", extent=(-9.5, 10.5, 9.5, -9.5))
    for dx, colour, label, at, align in ((0, ORANGE, "A", -4.5, "left"), (1, "#00e0ff", "B", 5.5, "right")):
        axes[0].add_patch(Rectangle((dx - 4.5, -4.5), 9, 9, fill=False, ec=colour, lw=2.2))
        axes[0].add_patch(Rectangle((dx - 0.5, -0.5), 1, 1, fc=colour, ec="none"))
        axes[0].text(at, -5.0, label, color=colour, ha=align, va="bottom", fontsize=12, fontweight="bold")
    axes[0].set_title("čtverce 9 × 9 sousedů A a B", fontsize=11)
    axes[1].imshow(np.clip(box_mean(img, 8), 0, 1))
    axes[1].set_title("průměr čtverce 17 × 17", fontsize=11)
    for ax in axes:
        bare(ax)
    save(fig, "idea_box.png")
    L = lightness(np.dstack([img, np.ones((h, w), np.float32)]))
    p = np.pad(L, 1, mode="edge")
    around = (sum(p[1 + j:1 + j + h, 1 + i:1 + i + w] for j in (-1, 0, 1) for i in (-1, 0, 1)) - L) / 8
    idea("idea_neighbours.png", [("světlost", L), ("rozdíl od průměru sousedů, zesílený 20krát", 20 * np.abs(L - around))],
         width=3.4)
    grainy = noisy(img)
    strength = sobel_strength(oklab_of(box_mean(grainy, 2))[..., 0])
    mask = np.clip(4 * strength, 0, 1)[..., None]
    smooth = linear_to_srgb(box_mean(srgb_to_linear(grainy.astype(np.float64)), 4))
    idea("idea_mask.png", [("zašuměná", grainy), ("hrany jako maska", mask[..., 0]),
                           ("rozmazaná mimo hrany", mask * grainy + (1 - mask) * smooth)])
    r = 3
    pad = np.pad(grainy, ((r, r), (r, r), (0, 0)), mode="edge")
    stack = np.stack([pad[r + dy:r + dy + h, r + dx:r + dx + w] for dy in range(-r, r + 1) for dx in range(-r, r + 1)])
    idea("idea_median.png", [("zašuměná", grainy), ("medián okolí 7 × 7", np.median(stack, axis=0))])
    idea("idea_similar.png", [("zašuměná", grainy), ("jen sousedé podobné barvy", bilateral_np(grainy, 3.0, 0.1))])


# --- Section 2's colour spaces, on a yellow sun in a blue sky --------------------------
SUN, SKY = np.array([1.0, 0.8, 0.05]), np.array([0.05, 0.3, 0.95])


def sun_picture(side=200, radius=70.5):
    yy, xx = np.mgrid[0:side, 0:side]
    inside = np.clip(radius - np.hypot(xx - side / 2, yy - side / 2), 0, 1)[..., None]
    return inside * SUN + (1 - inside) * SKY


SPACES = {"sRGB": (lambda c: c, lambda c: np.clip(c, 0, 1)),
          "lineární RGB": (srgb_to_linear, lambda c: linear_to_srgb(np.clip(c, 0, 1))),
          "OKLab": (lambda c: to_oklab(srgb_to_linear(c)), lambda c: linear_to_srgb(np.clip(oklab_to_linear(c), 0, 1)))}


def sharpened(img, name, radius=4, amount=2.0):
    """The SPACES kernel's sharpen: in OKLab only L, elsewhere every channel."""
    into, back = SPACES[name]
    c = into(img)
    sharp = c + amount * (c - box_mean(c, radius))
    if name == "OKLab":
        sharp = np.dstack([sharp[..., 0], c[..., 1:]])
    return back(sharp)


def spaces():
    """The sun blurred and sharpened by the same mean in sRGB, linear RGB and OKLab, sharpen in OKLab on L only."""
    img = sun_picture()
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.6), gridspec_kw={"wspace": 0.04, "hspace": 0.12})
    for col, (name, (into, back)) in enumerate(SPACES.items()):
        axes[0, col].imshow(back(box_mean(into(img), 8)))
        axes[0, col].set_title(name, fontsize=12)
        axes[1, col].imshow(sharpened(img, name))
    axes[1, 2].set_title("jen světlost L", fontsize=10, color=GREY)
    axes[0, 0].set_ylabel("rozmazání", fontsize=12)
    axes[1, 0].set_ylabel("zaostření", fontsize=12)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    save(fig, "spaces.png")


def spaces_split():
    """The sun's edge before and after sharpening, split into OKLab lightness L and chroma C."""
    img = sun_picture()
    window = (slice(60, 140), slice(120, 200))       # the sun's right edge and the sky beside it
    columns = [("původní", img), ("zaostření v sRGB", sharpened(img, "sRGB")),
               ("zaostření L v OKLabu", sharpened(img, "OKLab"))]
    fig, axes = plt.subplots(3, 3, figsize=(8.4, 8.6), gridspec_kw={"wspace": 0.05, "hspace": 0.12})
    for col, (title, picture) in enumerate(columns):
        lab = to_oklab(srgb_to_linear(picture))
        axes[0, col].imshow(picture[window])
        axes[0, col].set_title(title, fontsize=12)
        axes[1, col].imshow(lab[..., 0][window], cmap="gray", vmin=0.3, vmax=1.0)
        axes[2, col].imshow(np.hypot(lab[..., 1], lab[..., 2])[window], cmap="gray", vmin=0, vmax=0.35)
    for ax, label in zip(axes[:, 0], ("obrázek", "světlost L", "sytost C")):
        ax.set_ylabel(label, fontsize=12)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    save(fig, "spaces_split.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    make_photo()
    convolution_window()
    kernel_gallery()
    separable()
    edge_modes()
    shared_tile()
    sobel()
    bilateral()
    second_derivative()
    numpy_shifts()
    ideas()
    more_ideas()
    lens_pictures()
    box_window()
    spaces()
    spaces_split()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
