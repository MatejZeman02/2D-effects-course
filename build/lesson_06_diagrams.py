"""Draws the lesson 6 diagrams into imgs/6 of the lecture folder, and the HDR test picture.

rocket_hdr.npz is scikit-image's rocket photo (public domain) turned into
linear light, with everything near white made up to a hundred times brighter,
so the lamps on the pad are lights and not white paint. It is stored as half
floats, which is what a Sara layer keeps. The arithmetic of the pictures is
the notebook's own: the helpers, the curves and the solved tonemapper come
from the NumPy cells of lesson_06_notebook.py.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "imgs" / "6"
sys.path.insert(0, str(HERE))
import lesson_06_notebook as nb  # noqa: E402

ORANGE = "#F58518"
BLUE = "#4C78A8"
GREY = "#6B6B6B"

space = {"np": np}
for source in (nb.HELPERS_NP, nb.CURVES_NP, nb.SOLUTION_AUTO_NP, nb.SOLUTION_APPLY):
    exec(source, space)
srgb_to_linear, linear_to_srgb = space["srgb_to_linear"], space["linear_to_srgb"]
oklab, oklab_to_linear, luminance = space["oklab"], space["oklab_to_linear"], space["luminance"]
aces, reinhard, blur, auto_exposure = space["aces"], space["reinhard"], space["blur"], space["auto_exposure"]


def make_hdr():
    """rocket_hdr.npz: the rocket in linear light, its near whites boosted up to a hundred times."""
    import skimage.data
    rgb = skimage.data.rocket().astype(np.float32) / 255
    lin = srgb_to_linear(rgb)
    t = np.clip((luminance(lin) - 0.8) / 0.2, 0, 1)
    t = t * t * (3 - 2 * t)                 # smoothstep, so the boost starts gently
    hdr = lin * (1 + 100 * t ** 3)[..., None]
    np.savez_compressed(OUT / "rocket_hdr.npz", rgb=hdr.astype(np.float16))


def load_hdr():
    return np.load(OUT / "rocket_hdr.npz")["rgb"].astype(np.float32)


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


def exposure_strip(hdr):
    fig, axes = plt.subplots(1, 5, figsize=(15, 2.3), gridspec_kw={"wspace": 0.03})
    for ax, stops in zip(axes, (-4, -2, 0, 2, 4)):
        ax.imshow(linear_to_srgb(np.clip(hdr * 2.0 ** stops, 0, 1)))
        ax.set_title(f"{stops:+d} EV" if stops else "0 EV", fontsize=11)
        bare(ax)
    save(fig, "exposure_strip.png")


def dynamic_range(hdr):
    """The histogram of log2 luminance, the screen's eight stops and the geometric mean."""
    y = luminance(hdr).ravel()
    stops = np.log2(1e-4 + y)
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.hist(stops, bins=120, color=BLUE, alpha=0.85)
    ax.set_yscale("log")
    ax.axvspan(-8, 0, color=ORANGE, alpha=0.12)
    ax.set_ylim(0.6, 2e6)
    ax.text(-4, 3e5, "obrazovka, 8 EV", ha="center", color=ORANGE, fontsize=11)
    mean = np.log2(np.exp(np.log(1e-4 + y).mean()))
    ax.axvline(mean, color="black", lw=1.5, ls="--")
    ax.text(mean + 0.2, 2.5e4, "geometrický průměr", ha="left", fontsize=10)
    ax.set_xlabel("$\\log_2$ jasu (EV, 0 je bílá)", fontsize=11)
    ax.set_ylabel("počet pixelů", fontsize=11)
    plain(ax)
    save(fig, "dynamic_range.png")


def tone_curves():
    x = np.logspace(-2, 2, 500)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, np.clip(x, 0, 1), color=GREY, lw=2, label="ořezání")
    ax.plot(x, reinhard(x), color=BLUE, lw=2.2, label="Reinhard, $x / (1 + x)$")
    ax.plot(x, aces(x), color=ORANGE, lw=2.4, label="ACES (Narkowicz)")
    ax.set_xscale("log")
    ax.set_xlabel("světlo ve scéně (log)", fontsize=11)
    ax.set_ylabel("hodnota na obrazovce", fontsize=11)
    ax.axhline(1, color="#cccccc", lw=1, zorder=0)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    plain(ax)
    save(fig, "tone_curves.png")


def reduce_tree():
    """Sixteen numbers summed in four steps, half the threads working in each."""
    rng = np.random.default_rng(6)
    values = rng.integers(1, 9, 16)
    levels = [values]
    while len(levels[-1]) > 1:
        v = levels[-1]
        levels.append(v[0::2] + v[1::2])
    fig, ax = plt.subplots(figsize=(12, 4.2))
    for depth, v in enumerate(levels):
        width = 16 / len(v)
        for i, value in enumerate(v):
            x = i * width + width / 2
            y = -depth * 1.3
            ax.add_patch(Rectangle((x - 0.42, y - 0.4), 0.84, 0.8, fc=ORANGE if depth == len(levels) - 1 else "#dbe7f3",
                                   ec=GREY, lw=0.8))
            ax.text(x, y, str(value), ha="center", va="center", fontsize=10)
            if depth:
                for child in (2 * i, 2 * i + 1):
                    cx = child * width / 2 + width / 4
                    ax.add_patch(FancyArrowPatch((cx, y + 1.3 - 0.4), (x, y + 0.4), arrowstyle="-|>",
                                                 mutation_scale=8, color=BLUE, lw=0.8))
        label = "16 pixelů" if depth == 0 else f"krok {depth}: {len(v)} " + ("vlákno" if len(v) == 1 else "vlákna" if len(v) < 5 else "vláken")
        ax.text(16.6, -depth * 1.3, label, va="center", fontsize=10, color=GREY)
    ax.set_xlim(-0.3, 19.5)
    ax.set_ylim(-len(levels) * 1.3 + 0.6, 0.7)
    bare(ax)
    save(fig, "reduce.png")


def bloom_steps(hdr):
    e = auto_exposure(hdr)
    y = luminance(hdr)
    bright = hdr * (np.maximum(y - nb.THRESHOLD, 0) / np.maximum(y, 1e-4))[..., None]
    glow = blur(bright, nb.SIGMA)
    panels = [(aces(hdr * e), "tónová křivka"), (aces(bright * e), f"světlo nad prahem {nb.THRESHOLD:g}"),
              (aces(glow * e), "rozmazané"),
              (aces((hdr + nb.STRENGTH * glow) * e), "se září")]
    crop = (slice(230, 427), slice(40, 600))
    fig, axes = plt.subplots(2, 2, figsize=(13, 5.0), gridspec_kw={"wspace": 0.03, "hspace": 0.16})
    for ax, (picture, title) in zip(axes.ravel(), panels):
        ax.imshow(linear_to_srgb(picture[crop]))
        ax.set_title(title, fontsize=11)
        bare(ax)
    save(fig, "bloom_steps.png")


def fit_numpy(c):
    """The notebook's check cell, for an (..., 3) array."""
    lab = oklab(c)
    lab[..., 0] = np.minimum(lab[..., 0], 1.0)
    lo, hi = np.zeros(c.shape[:-1]), np.ones(c.shape[:-1])
    for _ in range(16):
        mid = 0.5 * (lo + hi)
        rgb = oklab_to_linear(np.concatenate([lab[..., :1], mid[..., None] * lab[..., 1:]], axis=-1))
        ok = np.all((rgb >= 0) & (rgb <= 1), axis=-1)
        lo, hi = np.where(ok, mid, lo), np.where(ok, hi, mid)
    out = np.clip(oklab_to_linear(np.concatenate([lab[..., :1], lo[..., None] * lab[..., 1:]], axis=-1)), 0, 1)
    inside = np.all((c >= 0) & (c <= 1), axis=-1)
    out[inside] = c[inside]
    return out


def gamut():
    """Saturated colours of every hue at rising exposure, by the three modes of the Gamut kernel."""
    hues = np.linspace(0, 2 * np.pi, 7)[:-1]
    lab = np.stack([np.full(6, 0.6), 0.2 * np.cos(hues), 0.2 * np.sin(hues)], axis=-1)
    base = np.clip(oklab_to_linear(lab), 0, None)
    exposure = 2.0 ** np.linspace(-3, 6, 240)
    c = base[:, None, :] * exposure[None, :, None]                     # (hue, exposure, 3)
    y = luminance(c)
    scaled = c * (aces(y) / np.maximum(y, 1e-4))[..., None]
    rows = [(aces(c), "křivka po složkách"), (np.clip(scaled, 0, 1), "jas, pak ořezání"),
            (fit_numpy(scaled), "jas, pak sytost v OKLabu")]
    fig, axes = plt.subplots(3, 1, figsize=(10, 4.6), gridspec_kw={"hspace": 0.75})
    for ax, (picture, title) in zip(axes, rows):
        ax.imshow(linear_to_srgb(np.repeat(picture, 8, axis=0)), aspect="auto", extent=(-3, 6, 6, 0))
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_yticks([])
        plain(ax)
        ax.spines["left"].set_visible(False)
    axes[-1].set_xlabel("expozice (EV)", fontsize=11)
    save(fig, "gamut.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    make_hdr()
    hdr = load_hdr()
    exposure_strip(hdr)
    dynamic_range(hdr)
    tone_curves()
    reduce_tree()
    bloom_steps(hdr)
    gamut()
    print("wrote", sorted(p.name for p in OUT.glob("*.*")))


if __name__ == "__main__":
    main()
