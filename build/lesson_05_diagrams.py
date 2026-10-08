"""Draws the lesson 5 diagrams into imgs/5 of the lecture folder, and the test photo.

chelsea.png is scikit-image's cat photo (CC0) enlarged twice with Lanczos.
The arithmetic is the notebook's own: the colour conversions, blur, xdog and
toon come from the NumPy cells of lesson_05_notebook.py, so a picture shows
what the cells compute.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "imgs" / "5"
sys.path.insert(0, str(HERE))
import lesson_05_notebook as nb  # noqa: E402

ORANGE = "#F58518"
BLUE = "#4C78A8"
GREY = "#6B6B6B"

space = {"np": np}
for source in (nb.HELPERS_NP, nb.TOON_NP, nb.SOLUTION_XDOG_NP):
    exec(source, space)
srgb_to_linear, linear_to_srgb, lightness = space["srgb_to_linear"], space["linear_to_srgb"], space["lightness"]
xdog, toon = space["xdog"], space["toon"]
DEFAULT = (nb.SIGMA, nb.P, nb.EPSILON, nb.PHI)


def make_photo():
    """chelsea.png: scikit-image's cat, 451 x 300, enlarged to the lesson's canvas."""
    import skimage.data
    small = Image.fromarray(skimage.data.chelsea())
    small.resize((nb.WIDTH, nb.HEIGHT), Image.LANCZOS).save(OUT / "chelsea.png", optimize=True)


def load(name):
    """A picture of imgs/5 as (h, w, 4) floats from 0 to 1, as Sara's import reads it."""
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


def plain(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def dog_profile():
    """Two Gaussians, their difference, and what the difference makes of a step."""
    x = np.linspace(-8, 8, 801)
    s, k = 1.5, nb.K
    g1 = np.exp(-x ** 2 / (2 * s ** 2)) / (s * np.sqrt(2 * np.pi))
    g2 = np.exp(-x ** 2 / (2 * (k * s) ** 2)) / (k * s * np.sqrt(2 * np.pi))
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"wspace": 0.25})
    ax = axes[0]
    ax.plot(x, g1, color=BLUE, lw=2, label="$G_\\sigma$")
    ax.plot(x, g2, color=GREY, lw=2, label="$G_{k\\sigma}$")
    ax.plot(x, g1 - g2, color=ORANGE, lw=2.6, label="$G_\\sigma - G_{k\\sigma}$")
    ax.axhline(0, color="#bbbbbb", lw=1, zorder=0)
    ax.set_title("jádro: rozdíl dvou Gaussů", fontsize=12)
    ax.legend(frameon=False, fontsize=11)
    plain(ax)
    ax = axes[1]
    step = np.where(x < 0, 0.2, 0.8)
    # a blurred step is the step times the Gaussian's integral, so D is a difference of two erfs
    from math import erf
    cdf = np.vectorize(lambda t: 0.5 * (1 + erf(t / np.sqrt(2))))
    dog = 0.6 * (cdf(x / s) - cdf(x / (k * s)))
    ax.plot(x, step, color=GREY, lw=2, label="světlost $L$ (hrana)")
    ax.plot(x, 0.5 + 4 * dog, color=ORANGE, lw=2.6, label="$0.5 + 4D$")
    ax.axhline(0.5, color="#bbbbbb", lw=1, zorder=0)
    ax.set_title("odezva na hranu", fontsize=12)
    ax.legend(frameon=False, fontsize=11, loc="upper left")
    ax.set_xlabel("pozice (pixely)", fontsize=11)
    plain(ax)
    save(fig, "dog_profile.png")


def xdog_curve():
    """The soft threshold T(S) for a few phi, with epsilon marked."""
    s = np.linspace(0, 1, 600)
    eps = nb.EPSILON
    fig, ax = plt.subplots(figsize=(8, 4))
    for phi, colour in ((2, "#9ecae1"), (10, BLUE), (50, ORANGE), (500, "black")):
        t = np.where(s >= eps, 1.0, 1 + np.tanh(phi * (s - eps)))
        ax.plot(s, t, lw=2.2, color=colour, label=f"$\\varphi = {phi}$")
    ax.axvline(eps, color="#bbbbbb", lw=1, ls="--")
    ax.text(eps + 0.01, 0.05, "$\\varepsilon$", fontsize=13, color=GREY)
    ax.set_xlabel("$S$", fontsize=12)
    ax.set_ylabel("$T(S)$: 0 tuš, 1 papír", fontsize=11)
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    plain(ax)
    save(fig, "xdog_curve.png")


def xdog_params(L):
    """Six settings of the line art on the cat, the first the lesson's defaults."""
    settings = [DEFAULT, (1.5, 20, 0.6, 10), (2.0, 20, 0.5, 10),
                (1.5, 20, 0.5, 100), (1.5, 60, 0.3, 10), (1.0, 0, 0.55, 10)]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.4), gridspec_kw={"wspace": 0.03, "hspace": 0.14})
    for ax, (s, p, eps, phi) in zip(axes.ravel(), settings):
        ax.imshow(np.clip(xdog(L, s, p, eps, phi), 0, 1), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"σ {s:g}, p {p:g}, ε {eps:g}, φ {phi:g}", fontsize=11)
        bare(ax)
    axes[0, 0].set_title(axes[0, 0].get_title() + " (výchozí)", fontsize=11, color=ORANGE)
    save(fig, "xdog_params.png")


def posterised(img, levels):
    return toon(img, np.ones(img.shape[:2]), levels)


def toon_steps(img, L):
    crop = (slice(80, 520), slice(150, 810))
    ink = xdog(L, *DEFAULT)
    panels = [(img[..., :3], "fotka"), (posterised(img, nb.LEVELS), f"posterizace, {nb.LEVELS} úrovní"),
              (np.dstack([ink] * 3), "čáry XDoG"), (toon(img, ink, nb.LEVELS), "toon: obojí")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 2.9), gridspec_kw={"wspace": 0.04})
    for ax, (picture, title) in zip(axes, panels):
        ax.imshow(np.clip(picture[crop], 0, 1))
        ax.set_title(title, fontsize=12)
        bare(ax)
    save(fig, "toon_steps.png")


def kuwahara_numpy(img, r):
    """The notebook's check cell, as a function."""
    lin = srgb_to_linear(img[..., :3])
    lum = lin @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    h, w = lum.shape
    P = np.pad(lin, ((r, r), (r, r), (0, 0)), mode="edge")
    Q = np.pad(lum, r, mode="edge")
    best, best_var = np.zeros_like(lin), np.full((h, w), np.inf)
    for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        total, s1, s2 = np.zeros_like(lin), np.zeros((h, w)), np.zeros((h, w))
        for dy in range(r + 1):
            for dx in range(r + 1):
                y, x = r + sy * dy, r + sx * dx
                total += P[y:y + h, x:x + w]
                s1 += Q[y:y + h, x:x + w]
                s2 += Q[y:y + h, x:x + w] ** 2
        n = (r + 1) ** 2
        var = s2 / n - (s1 / n) ** 2
        calmer = var < best_var
        best[calmer], best_var[calmer] = total[calmer] / n, var[calmer]
    return linear_to_srgb(best)


def kuwahara(img):
    """The four squares on a diagonal edge, the calmest one marked, and the filter on a crop of the photo."""
    n = 11
    y, x = np.mgrid[0:n, 0:n]
    edge = np.where(x + 0.6 * y > 8.5, 0.85, 0.25) + np.random.default_rng(5).normal(0, 0.04, (n, n))
    cy, cx, r = 5, 5, 3
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), gridspec_kw={"width_ratios": [1, 1.4, 1.4], "wspace": 0.06})
    ax = axes[0]
    ax.imshow(np.clip(edge, 0, 1), cmap="gray", vmin=0, vmax=1)
    variances = {}
    for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        xs = sorted((cx, cx + sx * r))
        ys = sorted((cy, cy + sy * r))
        block = edge[ys[0]:ys[1] + 1, xs[0]:xs[1] + 1]
        variances[(sx, sy)] = (block.var(), xs, ys)
    calmest = min(variances, key=lambda q: variances[q][0])
    for q, (var, xs, ys) in variances.items():
        chosen = q == calmest
        ax.add_patch(Rectangle((xs[0] - 0.5, ys[0] - 0.5), r + 1, r + 1, fill=False,
                               ec=ORANGE if chosen else BLUE, lw=3.2 if chosen else 1.4,
                               ls="-" if chosen else "--"))
    ax.add_patch(Rectangle((cx - 0.5, cy - 0.5), 1, 1, fc="none", ec="red", lw=2))
    ax.set_title("čtyři čtverce, nejklidnější oranžově", fontsize=11)
    bare(ax)
    crop = (slice(150, 330), slice(330, 600))
    piece = img[crop[0].start - 8:crop[0].stop + 8, crop[1].start - 8:crop[1].stop + 8]
    out = kuwahara_numpy(piece, nb.RADIUS)[8:-8, 8:-8]
    axes[1].imshow(img[crop][..., :3])
    axes[1].set_title("fotka", fontsize=12)
    axes[2].imshow(np.clip(out, 0, 1))
    axes[2].set_title(f"Kuwahara, poloměr {nb.RADIUS}", fontsize=12)
    for ax in axes[1:]:
        bare(ax)
    save(fig, "kuwahara.png")


def pipeline():
    """The lesson's layers as a chain, an arrow from each layer to the kernel that reads it."""
    boxes = {
        "chelsea": (0.0, 1.0, "#eeeeee", "fotka"),
        "Kuwahara": (2.4, 2.0, "#fde0c5", "sekce 4"),
        "Blurs": (2.4, 0.0, "#dbe7f3", "sekce 1"),
        "DoG": (4.8, -1.0, "#dbe7f3", "sekce 1"),
        "XDoG": (4.8, 0.6, "#dbe7f3", "sekce 2"),
        "Toon": (7.2, 1.3, "#fde0c5", "sekce 3"),
    }
    edges = [("chelsea", "Kuwahara", "-"), ("chelsea", "Blurs", "-"), ("Blurs", "DoG", "-"),
             ("Blurs", "XDoG", "-"), ("XDoG", "Toon", "-"), ("chelsea", "Toon", "-"),
             ("Kuwahara", "Toon", "--"), ("Kuwahara", "Blurs", "--")]
    fig, ax = plt.subplots(figsize=(12, 4.2))
    w, h = 1.6, 0.7
    for name, (x, y, colour, note) in boxes.items():
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=colour, ec=GREY, lw=1.2))
        ax.text(x + w / 2, y + h / 2 + 0.08, name, ha="center", va="center", fontsize=12, family="monospace")
        ax.text(x + w / 2, y + 0.14, note, ha="center", va="center", fontsize=9, color=GREY)
    for a, b, style in edges:
        xa, ya = boxes[a][0] + w, boxes[a][1] + h / 2
        xb, yb = boxes[b][0], boxes[b][1] + h / 2
        if a == "Kuwahara" and b == "Blurs":
            xa, ya, xb, yb = boxes[a][0] + w / 2, boxes[a][1], boxes[b][0] + w / 2, boxes[b][1] + h
        ax.add_patch(FancyArrowPatch((xa, ya), (xb, yb), arrowstyle="-|>", mutation_scale=14,
                                     color=ORANGE if style == "--" else GREY, lw=1.6, ls=style,
                                     connectionstyle="arc3,rad=0.0"))
    ax.text(4.0, -1.6, "plné šipky: výchozí řetěz, čárkované: varianta s malbou z konce sekce 4",
            ha="center", fontsize=10, color=GREY)
    ax.set_xlim(-0.3, 9.1)
    ax.set_ylim(-1.8, 3.0)
    ax.set_aspect("equal")
    bare(ax)
    save(fig, "pipeline.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    make_photo()
    img = load("chelsea.png")
    L = lightness(img)
    dog_profile()
    xdog_curve()
    xdog_params(L)
    toon_steps(img, L)
    kuwahara(img)
    pipeline()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
