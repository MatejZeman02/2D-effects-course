"""Draws the lesson 4 diagrams into imgs/4 of the lecture folder, and the blurred test photo.

The transforms are the notebook's own: spectrum, gaussian_filter, blur_fft and
the colour conversions come from the NumPy cells of lesson_04_notebook.py, so
a picture shows what the cells compute. astronaut_blur.png is astronaut.png
blurred by that blur_fft in linear light and rounded to eight bits, which the
notebook's Wiener filter then undoes.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "imgs" / "4"
sys.path.insert(0, str(HERE))
import lesson_04_notebook as nb  # noqa: E402

ORANGE = "#F58518"
BLUE = "#4C78A8"
GREY = "#6B6B6B"
N = nb.SIDE

space = {"np": np}
for source in (nb.HELPERS_NP, nb.SOLUTION_SPECTRUM, nb.SOLUTION_GAUSSIAN, nb.SOLUTION_BLUR):
    exec(source, space)
srgb_to_linear, linear_to_srgb, lightness = space["srgb_to_linear"], space["linear_to_srgb"], space["lightness"]
spectrum, gaussian_filter, blur_fft = space["spectrum"], space["gaussian_filter"], space["blur_fft"]


def load(name):
    """A picture of imgs/4 as (h, w, 4) floats from 0 to 1, as Sara's import reads it."""
    rgb = np.asarray(Image.open(OUT / name).convert("RGB"), np.float32) / 255
    return np.dstack([rgb, np.ones(rgb.shape[:2], np.float32)])


def blurred_photo(img):
    """*img*, (h, w, 3 or 4) in sRGB, blurred as astronaut_blur.png is: in linear light, to eight bits."""
    lin = srgb_to_linear(img[..., :3].astype(np.float64))
    out = np.dstack([blur_fft(lin[..., c], nb.BLUR_SIGMA) for c in range(3)])
    return np.round(np.clip(linear_to_srgb(out), 0, 1) * 255).astype(np.uint8)


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def bare(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def grey(ax, a, **keys):
    ax.imshow(a, cmap="gray", vmin=0, vmax=1, interpolation="antialiased", **keys)
    bare(ax)


def log_spectrum(a):
    """The notebook's spectrum of a grey picture, as one channel."""
    return spectrum(np.fft.fft2(a))[..., 0]


def box(width, height, angle=0.0, ss=4):
    """A white rectangle on black in the middle of the canvas, turned by *angle* degrees, antialiased."""
    s = (np.arange(N * ss) + 0.5) / ss - N / 2
    y, x = s[:, None], s[None, :]
    t = np.radians(angle)
    along, across = x * np.cos(t) + y * np.sin(t), -x * np.sin(t) + y * np.cos(t)
    inside = (np.abs(along) < width / 2) & (np.abs(across) < height / 2)
    return inside.reshape(N, ss, N, ss).mean(axis=(1, 3))


def waves():
    """Four basis waves and their spectra: three dots, the arrow (u, v) to one of them."""
    shown = [(4, 0), (0, 6), (6, 3), (10, -5)]
    y, x = np.mgrid[0:N, 0:N]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.3), gridspec_kw={"hspace": 0.28, "wspace": 0.2})
    for col, (u, v) in enumerate(shown):
        grey(axes[0, col], 0.5 + 0.5 * np.cos(2 * np.pi * (u * x + v * y) / N))
        axes[0, col].set_title(f"u = {u}, v = {v}", fontsize=12)
        ax = axes[1, col]
        ax.set_facecolor("black")
        ax.set_xlim(-12.5, 12.5)
        ax.set_ylim(12.5, -12.5)                          # v grows downwards, as y in a picture
        ax.set_aspect("equal")
        for k in range(-12, 13, 4):
            ax.axhline(k, color="#333333", lw=0.6, zorder=0)
            ax.axvline(k, color="#333333", lw=0.6, zorder=0)
        ax.scatter([0, u, -u], [0, v, -v], s=[90, 60, 60], color="white", zorder=3)
        ax.annotate("", xy=(u, v), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2, shrinkA=0, shrinkB=5), zorder=4)
        ax.set_xticks([-12, 0, 12])
        ax.set_yticks([-12, 0, 12])
        ax.tick_params(labelsize=9)
        ax.set_xlabel("u", fontsize=10, labelpad=1)
        if col == 0:
            ax.set_ylabel("v", fontsize=10, labelpad=1)
    fig.text(0.05, 0.71, "vlna", rotation=90, fontsize=12, va="center")
    fig.text(0.05, 0.29, "spektrum", rotation=90, fontsize=12, va="center")
    save(fig, "waves.png")


def shift_layout():
    """fft2's order of the coefficients, zero frequency in the corner, and fftshift's, zero in the middle."""
    colours = {"A": "#c6dbef", "B": "#fdd0a2", "C": "#c7e9c0", "D": "#dadaeb"}
    before = [["A", "B"], ["C", "D"]]
    after = [["D", "C"], ["B", "A"]]
    photo = lightness(load("astronaut.png"))
    F = np.fft.fft2(photo)
    raw = np.log1p(np.abs(F))
    fig, axes = plt.subplots(2, 2, figsize=(9, 9.4), gridspec_kw={"hspace": 0.16, "wspace": 0.1})
    for col, (layout, title, zero) in enumerate([(before, "výstup np.fft.fft2", (0.02, 0.02)),
                                                 (after, "po np.fft.fftshift", (0.5, 0.5))]):
        ax = axes[0, col]
        for r in range(2):
            for c in range(2):
                q = layout[r][c]
                ax.add_patch(plt.Rectangle((c * 0.5, r * 0.5), 0.5, 0.5, facecolor=colours[q], edgecolor="white", lw=2))
                ax.text(c * 0.5 + 0.25, r * 0.5 + 0.25, q, ha="center", va="center", fontsize=22, color=GREY)
        ax.scatter([zero[0]], [zero[1]], s=110, color=ORANGE, zorder=3, clip_on=False)
        ax.annotate("nulová frekvence", xy=zero, xytext=(zero[0] + 0.12, zero[1] + 0.13), fontsize=11,
                    color=ORANGE, arrowprops=dict(arrowstyle="-", color=ORANGE, lw=1.2))
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.set_aspect("equal")
        bare(ax)
        ax.set_title(title, fontsize=12, family="monospace")
    grey(axes[1, 0], raw / raw.max())
    grey(axes[1, 1], log_spectrum(photo))
    axes[1, 0].set_title("spektrum fotky bez posunutí", fontsize=11)
    axes[1, 1].set_title("spektrum fotky po posunutí", fontsize=11)
    fig.text(0.5, 0.505, "A: nezáporné u i v, B: záporné u, C: záporné v, D: obě záporné", ha="center", fontsize=10, color=GREY)
    save(fig, "fftshift.png")


def pattern(period=32, radius=8.0):
    """Soft discs on a square grid, *period* pixels apart, a pattern that repeats exactly."""
    y, x = (np.mgrid[0:N, 0:N] % period) - period / 2 + 0.5
    return np.clip(radius - np.hypot(x, y) + 0.5, 0, 1)


def pairs():
    """Pictures and their spectra: narrow against wide, a Gaussian, a turn, a pattern, a photo."""
    y, x = np.mgrid[0:N, 0:N] - N / 2 + 0.5
    shown = [("úzký obdélník", box(8, 96)),
             ("široký obdélník", box(32, 96)),
             ("Gauss", np.exp(-(x ** 2 + y ** 2) / (2 * 8.0 ** 2))),
             ("otočený obdélník", box(8, 96, angle=30)),
             ("opakovaný vzor", pattern()),
             ("fotka", lightness(load("astronaut.png")))]
    fig, axes = plt.subplots(2, len(shown), figsize=(15, 5.6), gridspec_kw={"hspace": 0.08, "wspace": 0.06})
    for col, (label, picture) in enumerate(shown):
        grey(axes[0, col], picture)
        seen = log_spectrum(picture)
        if label == "opakovaný vzor":
            # the dots are single pixels, which shrinking the picture would average away
            seen = seen.reshape(N // 4, 4, N // 4, 4).max(axis=(1, 3))
        grey(axes[1, col], seen)
        axes[0, col].set_title(label, fontsize=12)
    fig.text(0.115, 0.71, "obrázek", rotation=90, fontsize=12, va="center")
    fig.text(0.115, 0.29, "spektrum", rotation=90, fontsize=12, va="center")
    save(fig, "pairs.png")


def convolution():
    """The convolution theorem: a blur in the picture is a product in the spectrum."""
    sigma = 6.0
    photo = lightness(load("astronaut.png"))
    y, x = np.mgrid[0:N, 0:N] - N / 2
    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    blurred = blur_fft(photo, sigma)
    F = np.fft.fftshift(np.fft.fft2(photo))
    H = np.fft.fftshift(gaussian_filter(N, N, sigma))
    top = np.log1p(np.abs(F)).max()                         # one scale for both spectra
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.6), gridspec_kw={"hspace": 0.14, "wspace": 0.34})
    grey(axes[0, 0], photo)
    grey(axes[0, 1], kernel[N // 2 - 40:N // 2 + 40, N // 2 - 40:N // 2 + 40])
    grey(axes[0, 2], blurred)
    grey(axes[1, 0], np.log1p(np.abs(F)) / top)
    grey(axes[1, 1], H)
    grey(axes[1, 2], np.log1p(np.abs(F * H)) / top)
    for ax, title in zip(axes.ravel(), ["fotka $f$", "jádro $g$ (zvětšené)", "rozmazaná fotka $f * g$",
                                        "spektrum $F$", "spektrum jádra $H$", "$F \\cdot H$"]):
        ax.set_title(title, fontsize=12)
    for row, (sign, y_text) in enumerate([("∗", 0.705), ("×", 0.29)]):
        fig.text(0.365, y_text, sign, fontsize=28, ha="center", va="center")
        fig.text(0.655, y_text, "=", fontsize=28, ha="center", va="center")
    fig.text(0.08, 0.705, "konvoluce", rotation=90, fontsize=12, va="center")
    fig.text(0.08, 0.29, "násobení", rotation=90, fontsize=12, va="center")
    save(fig, "convolution.png")


def wiener_curves():
    """How much each filter amplifies a frequency: the blur, its inverse and Wiener at two K."""
    f = np.linspace(0, 0.5, 600)
    H = np.exp(-2 * np.pi ** 2 * nb.BLUR_SIGMA ** 2 * f ** 2)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.axhline(1, color="#bbbbbb", lw=1, zorder=0)
    ax.plot(f, H, color=GREY, lw=2.2, label="rozmazání $H$")
    ax.plot(f, 1 / H, color=GREY, lw=1.8, ls="--", label="dělení $1/H$")
    for K, colour, text in [(0.001 ** 2, BLUE, "$K = 0.001^2$"), (0.01 ** 2, ORANGE, "$K = 0.01^2$")]:
        ax.plot(f, H / (H ** 2 + K), color=colour, lw=2.2, label=f"Wiener, {text}")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 3e3)
    ax.set_xlim(0, 0.5)
    ax.set_xlabel("frekvence (periody na pixel)", fontsize=11)
    ax.set_ylabel("zesílení", fontsize=11)
    ax.text(0.497, 1.15, "beze změny", color="#999999", fontsize=9, va="bottom", ha="right")
    ax.set_title(f"Gaussovo rozmazání se $\\sigma = {nb.BLUR_SIGMA:g}$", fontsize=12)
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, "wiener.png")


def main():
    Image.fromarray(blurred_photo(load("astronaut.png"))).save(OUT / "astronaut_blur.png", optimize=True)
    waves()
    shift_layout()
    pairs()
    convolution()
    wiener_curves()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
