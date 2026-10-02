"""Draws the lesson 1 diagrams and its test picture into imgs/1 of the lecture folder."""
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

OUT = str(Path(__file__).resolve().parent.parent / "imgs" / "1")
os.makedirs(OUT, exist_ok=True)

ORANGE = "#F58518"
GREY = "#6B6B6B"


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# Colour conversions, the same as in the notebooks -----------------------------
def srgb_to_linear(c):
    a = np.abs(c)
    return np.sign(c) * np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    a = np.abs(c)
    return np.sign(c) * np.where(a <= 0.0031308, a * 12.92, 1.055 * a ** (1 / 2.4) - 0.055)


M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
               [0.2119034982, 0.6806995451, 0.1073969566],
               [0.0883024619, 0.2817188376, 0.6299787005]])
M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
               [1.9779984951, -2.4285922050, 0.4505937099],
               [0.0259040371, 0.7827717662, -0.8086757660]])


def linear_to_oklab(rgb):
    return np.cbrt(rgb @ M1.T) @ M2.T


def oklab_to_linear(lab):
    return ((lab @ np.linalg.inv(M2).T) ** 3) @ np.linalg.inv(M1).T


def mix(a, b, t):
    return (1 - t) * a + t * b


def blend(a, b, t, space):
    """Colours a and b (sRGB) mixed by t in one of three spaces, the result in sRGB."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    t = np.asarray(t, float)[..., None]
    if space == "srgb":
        return mix(a, b, t)
    if space == "linear":
        return linear_to_srgb(mix(srgb_to_linear(a), srgb_to_linear(b), t))
    lab = mix(linear_to_oklab(srgb_to_linear(a)), linear_to_oklab(srgb_to_linear(b)), t)
    return linear_to_srgb(oklab_to_linear(lab))


def hex_colour(text):
    return np.array([int(text[i:i + 2], 16) / 255 for i in (1, 3, 5)])


# 1. The test picture: three shaded balls under a sky ---------------------------
# Rendered in linear light and encoded to sRGB, so the shading is smooth and a
# posterise shows its bands plainly.
W, H = 1920, 1080
yy, xx = np.mgrid[0:H, 0:W] + 0.5
horizon = 0.6 * H
sky_t = np.clip(yy / horizon, 0, 1)[..., None] ** 0.8
sky = mix(np.array([0.16, 0.30, 0.66]), np.array([0.88, 0.72, 0.56]), sky_t)
ground_t = np.clip((yy - horizon) / (H - horizon), 0, 1)[..., None]
ground = mix(np.array([0.34, 0.29, 0.24]), np.array([0.07, 0.06, 0.05]), ground_t)
image = np.where((yy < horizon)[..., None], sky, ground)

balls = [  # centre x, centre y, radius, albedo in linear light
    (0.50 * W, 0.57 * H, 0.27 * H, np.array([0.05, 0.38, 0.33])),
    (0.21 * W, 0.67 * H, 0.20 * H, np.array([0.78, 0.17, 0.05])),
    (0.80 * W, 0.70 * H, 0.17 * H, np.array([0.28, 0.20, 0.76])),
]
for cx, cy, r, _ in balls:  # soft shadows on the ground
    d2 = ((xx - cx - 0.45 * r) / (1.15 * r)) ** 2 + ((yy - cy - 0.95 * r) / (0.20 * r)) ** 2
    image *= (1 - 0.65 * np.exp(-1.5 * d2))[..., None]

light = np.array([-0.45, -0.65, 0.62])
light /= np.linalg.norm(light)
half = light + np.array([0.0, 0.0, 1.0])
half /= np.linalg.norm(half)
sun = np.array([1.10, 1.03, 0.92])
for cx, cy, r, albedo in sorted(balls, key=lambda b: b[1] + b[2]):
    dx, dy = (xx - cx) / r, (yy - cy) / r
    inside = np.clip(1 - dx ** 2 - dy ** 2, 0, None)
    normal = np.stack([dx, dy, np.sqrt(inside)], axis=-1)
    diffuse = np.clip(normal @ light, 0, None)[..., None]
    ambient = (0.10 + 0.07 * np.clip(-dy, -1, 1))[..., None] * np.array([0.75, 0.85, 1.0])
    specular = 0.55 * np.clip(normal @ half, 0, None)[..., None] ** 60
    shaded = albedo * (sun * diffuse + ambient) + specular
    coverage = np.clip(r - np.hypot(xx - cx, yy - cy) + 0.5, 0, 1)[..., None]  # anti-aliased edge
    image = mix(image, shaded, coverage)

plt.imsave(os.path.join(OUT, "balls.png"), np.clip(linear_to_srgb(image), 0, 1))


# 2. One gradient mixed in three spaces ----------------------------------------
pairs = [("červená → zelená", [1, 0, 0], [0, 1, 0]),
         ("modrá → žlutá", [0, 0, 1], [1, 1, 0]),
         ("černá → bílá", [0, 0, 0], [1, 1, 1])]
spaces = [("srgb", "sRGB"), ("linear", "lineární RGB"), ("oklab", "OKLab")]
ramp = np.linspace(0, 1, 512)

fig, axes = plt.subplots(3, 3, figsize=(11, 2.9), gridspec_kw={"hspace": 0.18, "wspace": 0.06})
for col, (title, a, b) in enumerate(pairs):
    for row, (space, label) in enumerate(spaces):
        ax = axes[row, col]
        strip = np.clip(blend(a, b, ramp, space), 0, 1)
        ax.imshow(np.repeat(strip[None], 40, axis=0), aspect="auto", extent=(0, 1, 0, 1))
        ax.axvline(0.5, ymin=0, ymax=0.18, color="white", linewidth=1.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if row == 0:
            ax.set_title(title, fontsize=12)
        if col == 0:
            ax.set_ylabel(label, fontsize=11, rotation=0, ha="right", va="center")
save(fig, "mix_spaces.png")


# 3. A gradient map with three stops ---------------------------------------------
stops = [("dark", 0.0, hex_colour("#1b1035")), ("mid", 0.5, hex_colour("#c2412d")),
         ("light", 1.0, hex_colour("#ffe7a3"))]
t = np.linspace(0, 1, 768)
grey = np.clip(linear_to_srgb(oklab_to_linear(np.stack([t, 0 * t, 0 * t], axis=-1))), 0, 1)
low = t < 0.5
mapped = np.where(low[:, None], blend(stops[0][2], stops[1][2], np.clip(t / 0.5, 0, 1), "oklab"),
                  blend(stops[1][2], stops[2][2], np.clip((t - 0.5) / 0.5, 0, 1), "oklab"))

fig = plt.figure(figsize=(11, 2.6))
top = fig.add_axes([0.05, 0.62, 0.9, 0.24])
bottom = fig.add_axes([0.05, 0.16, 0.9, 0.24])
top.imshow(np.repeat(grey[None], 30, axis=0), aspect="auto", extent=(0, 1, 0, 1))
bottom.imshow(np.repeat(np.clip(mapped, 0, 1)[None], 30, axis=0), aspect="auto", extent=(0, 1, 0, 1))
for ax in (top, bottom):
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
top.set_xticks([0, 0.25, 0.5, 0.75, 1])
top.tick_params(labelsize=10, length=0)
top.set_title("světlost pixelu t = L", fontsize=12, loc="left")
bottom.set_xticks([])
for name, at, colour in stops:
    fig.add_artist(FancyArrowPatch((0.05 + 0.9 * at, 0.58), (0.05 + 0.9 * at, 0.43), transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=14, linewidth=1.8, color=GREY))
    bottom.text(at, -0.25, name, ha="center", va="top", fontsize=11, family="monospace",
                transform=bottom.transAxes)
bottom.text(0.25, 1.12, "barva z přechodu, míchaná v OKLabu", ha="center", va="bottom", fontsize=12,
            transform=bottom.transAxes)
save(fig, "gradient_map.png")


# 4. The Bayer matrix and what it does to a ramp ----------------------------------
BAYER = np.array([[0, 8, 2, 10],
                  [12, 4, 14, 6],
                  [3, 11, 1, 9],
                  [15, 7, 13, 5]])
fig = plt.figure(figsize=(11, 3.3))
mat = fig.add_axes([0.02, 0.1, 0.24, 0.8])
mat.imshow((BAYER + 0.5) / 16, cmap="gray", vmin=0, vmax=1)
for j in range(4):
    for i in range(4):
        value = BAYER[j, i]
        mat.text(i, j, str(value), ha="center", va="center", fontsize=14,
                 color="white" if value < 8 else "black")
mat.set_xticks(range(4))
mat.set_yticks(range(4))
mat.tick_params(length=0, labelsize=10)
mat.set_xlabel("x mod 4", fontsize=11)
mat.set_ylabel("y mod 4", fontsize=11)
mat.set_title("práh = (B + 0.5) / 16", fontsize=12)

ramp = np.tile(np.linspace(0, 1, 128), (16, 1))
ry, rx = np.mgrid[0:16, 0:128]
rows = [("zaokrouhlení, práh 0.5 všude", np.floor(ramp + 0.5)),
        ("Bayer 4 × 4", np.floor(ramp + (BAYER[ry % 4, rx % 4] + 0.5) / 16))]
for k, (label, img) in enumerate(rows):
    ax = fig.add_axes([0.33, 0.62 - k * 0.4, 0.64, 0.24])
    ax.imshow(np.clip(img, 0, 1), cmap="gray", vmin=0, vmax=1, interpolation="nearest", aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(label, fontsize=11, loc="left")
fig.text(0.33, 0.1, "přechod od černé po bílou, převedený na dvě úrovně", fontsize=10, color=GREY)
save(fig, "bayer.png")


# 5. Other thresholds: white noise and interleaved gradient noise ------------------
def fract(x):
    return x - np.floor(x)


def hash21(x, y):
    """sa_hash21 from Sara's noise library, white noise from 0 to 1."""
    q = fract(np.stack([x, y, x], axis=-1).astype(np.float32) * np.float32(0.1031))
    q += (q * (q[..., [1, 2, 0]] + np.float32(33.33))).sum(axis=-1, keepdims=True)
    return fract((q[..., 0] + q[..., 1]) * q[..., 2])


def ign(x, y):
    """Interleaved gradient noise, Jimenez 2014."""
    return fract(52.9829189 * fract(0.06711056 * x + 0.00583715 * y))


S = 128
gy, gx = np.mgrid[0:S, 0:S]
thresholds = [("práh 0.5", np.full((S, S), 0.5)),
              ("Bayer 4 × 4", (BAYER[gy % 4, gx % 4] + 0.5) / 16),
              ("bílý šum", hash21(gx, gy)),
              ("IGN", ign(gx, gy))]
# A soft glow on black, the case where banding shows worst.
glow = 0.5 * np.exp(-3.2 * (np.hypot(gx - S / 2 + 0.5, gy - S / 2 + 0.5) / (S / 2)) ** 2)
levels = 7
fig, axes = plt.subplots(2, 4, figsize=(11, 5.9), gridspec_kw={"hspace": 0.08, "wspace": 0.05})
for col, (label, d) in enumerate(thresholds):
    axes[0, col].imshow(d[:24, :24], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[1, col].imshow(np.floor(glow * levels + d) / levels, cmap="gray", vmin=0, vmax=1,
                        interpolation="nearest")
    axes[0, col].set_title(label, fontsize=12)
    for ax in axes[:, col]:
        ax.set_xticks([])
        ax.set_yticks([])
axes[0, 0].set_ylabel("práh d\n24 × 24 pixelů", fontsize=11)
axes[1, 0].set_ylabel("záře na tmavém pozadí\nna osmi úrovních", fontsize=11)
save(fig, "noise.png")


# 6. Why Floyd-Steinberg runs pixel after pixel -----------------------------------
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.set_xlim(-0.6, 7.6)
ax.set_ylim(4.1, -1.0)
ax.axis("off")
cx, cy = 3, 1
for j in range(4):
    for i in range(8):
        done = j < cy or (j == cy and i < cx)
        face = ORANGE if (i, j) == (cx, cy) else ("#D9D9D9" if done else "white")
        ax.add_patch(Rectangle((i - 0.5, j - 0.5), 1, 1, facecolor=face, edgecolor=GREY, linewidth=1))
for (i, j), weight in {(cx + 1, cy): "7/16", (cx - 1, cy + 1): "3/16", (cx, cy + 1): "5/16",
                       (cx + 1, cy + 1): "1/16"}.items():
    ax.add_patch(FancyArrowPatch((cx, cy), (i, j), arrowstyle="-|>", mutation_scale=16, linewidth=2,
                                 color="#222", shrinkA=8, shrinkB=12))
    ax.text(i, j + 0.28, weight, ha="center", va="center", fontsize=11)
ax.text(-0.5, -0.75, "šedé pixely jsou hotové, oranžový se právě zaokrouhluje, bílé čekají", fontsize=11, color="#333")
save(fig, "floyd_steinberg.png")

print("written to", OUT)
