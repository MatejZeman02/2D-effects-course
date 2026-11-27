"""Draws the lesson 2 diagrams into imgs/2 of the lecture folder.

The noise is the notebook's own: the hash, the value noise and the stops come
from the NumPy cells of lesson_02_notebook.py, so a picture shows what the
kernels compute.
"""
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

HERE = Path(__file__).resolve().parent
OUT = str(HERE.parent / "imgs" / "2")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, str(HERE))
import lesson_02_notebook as nb  # noqa: E402

ORANGE = "#F58518"
GREY = "#6B6B6B"

space = {"np": np}
for source in (nb.HELPERS_NP, nb.SOLUTION_VALUE_NP, nb.FBM_NP):
    exec(source, space)
random1, value_noise, fbm = space["random1"], space["value_noise"], space["fbm"]
cell_hash, to_lab, lab_to_linear = space["cell_hash"], space["to_lab"], space["lab_to_linear"]
linear_to_srgb, STOP_AT, STOP_LAB = space["linear_to_srgb"], space["STOP_AT"], space["STOP_LAB"]
srgb_to_linear = space["srgb_to_linear"]


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def bare(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def random2(cx, cy):
    """random2 of the kernel, two numbers from 0 to 1 a cell."""
    h = cell_hash(cx, cy)
    return (h & 65535).astype(np.float64) / 65536, (h >> 16).astype(np.float64) / 65536


def smooth(f):
    return f * f * (3 - 2 * f)


# 0. The brainstorm's ideas, each a height map and the terrain the lesson makes of it.
# The terrain is the notebook's own apply, handed the height map in place of its fBm.
# Every height map is moved and stretched to the mean and spread of the goal's fBm,
# so the sea covers as much, and the relief follows the size of the hills, as the
# goal's 400 at a scale of 400 px.
mix = space["mix"]
apply_space = dict(space)
exec(nb.SOLUTION_APPLY_NP, apply_space)
_goal = fbm((np.arange(1024)[None, :] + 0.5) / 400, (np.arange(640)[:, None] + 0.5) / 400, 6)
GOAL_MEAN, GOAL_SPREAD = float(_goal.mean()), float(_goal.std())
IDEA_H, IDEA_W, IDEA_CELL = 160, 256, 48
iy, ix = np.mgrid[0:IDEA_H, 0:IDEA_W]
px, py = ix + 0.5, iy + 0.5


def like_goal(h):
    return GOAL_MEAN + (h - h.mean()) / h.std() * GOAL_SPREAD


def terrain(height, relief=float(IDEA_CELL)):
    apply_space["fbm"] = lambda x, y, octaves: height
    img = np.ones(height.shape + (4,), np.float32)
    return np.clip(apply_space["apply"](img, 400.0, 6, 0.5, relief, 135.0)[..., :3], 0, 1)


def idea(name, panels):
    """One row of (title, picture) panels, a 2D picture shown as a height map, stretched."""
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 2.35), gridspec_kw={"wspace": 0.05})
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None, interpolation="nearest")
        ax.set_title(title, fontsize=11)
        bare(ax)
    save(fig, name)


def blur(img, sigma):
    fy, fx = np.fft.fftfreq(img.shape[0])[:, None], np.fft.fftfreq(img.shape[1])[None, :]
    return np.real(np.fft.ifft2(np.fft.fft2(img) * np.exp(-2 * (np.pi * sigma) ** 2 * (fx * fx + fy * fy))))


def waves(count, shortest, longest, seed):
    """*count* sines of lengths from *shortest* to *longest* px in random directions, each as tall as long."""
    rng = np.random.default_rng(seed)
    total = np.zeros((IDEA_H, IDEA_W))
    for _ in range(count):
        length = np.exp(rng.uniform(np.log(shortest), np.log(longest)))
        angle, phase = rng.uniform(0, 2 * np.pi, 2)
        total += length * np.sin(2 * np.pi / length * (np.cos(angle) * px + np.sin(angle) * py) + phase)
    return like_goal(total)


def hills(count, seed):
    """*count* round hills at random places, each as tall as wide."""
    rng = np.random.default_rng(seed)
    total = np.zeros((IDEA_H, IDEA_W))
    for _ in range(count):
        r, cx, cy = rng.uniform(8, 22), rng.uniform(0, IDEA_W), rng.uniform(0, IDEA_H)
        total += r * np.exp(-((px - cx) ** 2 + (py - cy) ** 2) / (2 * r * r))
    return like_goal(total)


def perlin(x, y):
    """Perlin's gradient noise: a random direction at each corner, its slope g · (p - corner) mixed."""
    cx, cy = np.floor(x), np.floor(y)
    fx, fy = x - cx, y - cy
    cx, cy = cx.astype(np.int64), cy.astype(np.int64)

    def corner(dx, dy):
        a = 2 * np.pi * random1(cx + dx, cy + dy)
        return np.cos(a) * (fx - dx) + np.sin(a) * (fy - dy)

    ux, uy = (f ** 3 * (f * (6 * f - 15) + 10) for f in (fx, fy))
    return mix(mix(corner(0, 0), corner(1, 0), ux), mix(corner(0, 1), corner(1, 1), ux), uy)


def diamond_square(levels, roughness, seed):
    rng = np.random.default_rng(seed)
    size = 2 ** levels + 1
    h = np.zeros((size, size))
    h[::size - 1, ::size - 1] = rng.random((2, 2))
    step, amplitude = size - 1, 1.0
    while step > 1:
        half = step // 2
        for y in range(half, size, step):                  # diamond: the middle of each square
            for x in range(half, size, step):
                h[y, x] = h[y - half:y + half + 1:step, x - half:x + half + 1:step].mean() + rng.uniform(-amplitude, amplitude)
        for y in range(0, size, half):                     # square: the middle of each edge
            for x in range((y + half) % step, size, step):
                around = [h[y + dy, x + dx] for dy, dx in ((-half, 0), (half, 0), (0, -half), (0, half))
                          if 0 <= y + dy < size and 0 <= x + dx < size]
                h[y, x] = np.mean(around) + rng.uniform(-amplitude, amplitude)
        step, amplitude = half, amplitude * roughness
    return h


HEIGHT, LAND = "výšková mapa", "krajina"
white = like_goal(random1(ix + 9000, iy + 9000).astype(np.float64))
idea("idea_white.png", [(HEIGHT, white), (LAND, terrain(white, 400.0))])
blurred = like_goal(blur(white, 10))
idea("idea_blur.png", [(HEIGHT, blurred), (LAND, terrain(blurred))])
few, many = waves(3, 90, 180, 4), waves(40, 6, 300, 5)
idea("idea_waves.png", [("3 vlny", few), (LAND, terrain(few)), ("40 vln", many), (LAND, terrain(many))])
bumps = hills(16, 11)
idea("idea_hills.png", [(HEIGHT, bumps), (LAND, terrain(bumps))])
gradient_noise = like_goal(perlin(px / IDEA_CELL + 3, py / IDEA_CELL + 5))
idea("idea_perlin.png", [(HEIGHT, gradient_noise), (LAND, terrain(gradient_noise))])
squares = like_goal(diamond_square(8, 0.5, 6)[:IDEA_H, :IDEA_W])
idea("idea_diamond.png", [(HEIGHT, squares), (LAND, terrain(squares))])

# random numbers only where the grid lines cross, a smooth blend between, and its terrain,
# over whole cells, so the outer crossings lie on the picture's edges
LX, LY, LW, LH = 11, 2, 5, 3                               # the first crossing's cell, cells across and down
LAT_W, LAT_H = LW * IDEA_CELL, LH * IDEA_CELL
values = value_noise(px[:LAT_H, :LAT_W] / IDEA_CELL + LX, py[:LAT_H, :LAT_W] / IDEA_CELL + LY)
fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.35), gridspec_kw={"wspace": 0.08})
axes[0].imshow(np.ones((LAT_H, LAT_W)), cmap="gray", vmin=0, vmax=1)
axes[1].imshow(values, cmap="gray", interpolation="nearest")
axes[2].imshow(terrain(like_goal(values)), interpolation="nearest")
for ax in axes[:2]:
    line = dict(color="#999" if ax is axes[0] else "#5dade2", linewidth=0.8)
    for k in range(LW + 1):
        ax.plot([k * IDEA_CELL - 0.5] * 2, [-0.5, LAT_H - 0.5], **line)
    for k in range(LH + 1):
        ax.plot([-0.5, LAT_W - 0.5], [k * IDEA_CELL - 0.5] * 2, **line)
lo, hi = values.min(), values.max()
for kx in range(LW + 1):
    for ky in range(LH + 1):
        v = float(random1(np.array([LX + kx]), np.array([LY + ky]))[0])
        at = (kx * IDEA_CELL - 0.5, ky * IDEA_CELL - 0.5)
        for ax in axes[:2]:
            ax.plot(*at, "o", ms=7, mfc=str(np.clip((v - lo) / (hi - lo), 0, 1)), mec=ORANGE, mew=1.6, clip_on=False, zorder=3)
        axes[0].text(at[0], at[1] - 7, f"{v:.2f}", fontsize=8, ha="center", va="bottom", color="#222",
                     bbox=dict(fc="white", ec="none", pad=0.6))
for ax, title in zip(axes, ("náhodná čísla na křižovatkách", "plynulý přechod mezi nimi", LAND)):
    ax.set_xlim(-0.5 - 16, LAT_W - 0.5 + 16)        # room for the outer crossings and their numbers
    ax.set_ylim(LAT_H - 0.5 + 8, -0.5 - 24)
    ax.set_title(title, fontsize=11)
    bare(ax)
save(fig, "idea_lattice.png")


# 1. Value noise: cells, a straight blend and a smooth one -------------------------
CELLS = 7
x = np.linspace(0, CELLS, 1400, endpoint=False)
cx = np.floor(x).astype(np.int64)
corners = random1(np.arange(CELLS + 1), np.zeros(CELLS + 1, np.int64))
f = x - cx
curves = [("čtverce (Cells)", corners[cx]),
          ("lineární přechod (Linear)", corners[cx] + (corners[cx + 1] - corners[cx]) * f),
          ("hladký přechod (Smooth)", corners[cx] + (corners[cx + 1] - corners[cx]) * smooth(f))]
S = 32                                                     # pixels a cell in the tiles
ty, tx = np.mgrid[0:4 * S, 0:CELLS * S]
px, py = (tx + 0.5) / S, (ty + 0.5) / S
gx, gy = np.floor(px).astype(np.int64), np.floor(py).astype(np.int64)
fx, fy = px - gx, py - gy


def tile(weight):
    a, b = random1(gx, gy), random1(gx + 1, gy)
    c, d = random1(gx, gy + 1), random1(gx + 1, gy + 1)
    wx, wy = weight(fx), weight(fy)
    return (a + (b - a) * wx) * (1 - wy) + (c + (d - c) * wx) * wy


tiles = [random1(gx, gy), tile(lambda f: f), tile(smooth)]
fig, axes = plt.subplots(2, 3, figsize=(12, 4.6), gridspec_kw={"height_ratios": [1, 1.05], "hspace": 0.25, "wspace": 0.06})
for col, ((title, curve), img) in enumerate(zip(curves, tiles)):
    ax = axes[0, col]
    ax.plot(x, curve, color="#222", linewidth=1.6)
    ax.plot(np.arange(CELLS + 1), corners, "o", color=ORANGE, markersize=6, zorder=3, clip_on=False)
    for k in range(CELLS + 1):
        ax.axvline(k, color="#DDD", linewidth=0.8, zorder=0)
    ax.set_xlim(0, CELLS)
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 1])
    ax.set_xticks([])
    ax.tick_params(labelsize=9, length=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(title, fontsize=12)
    axes[1, col].imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    bare(axes[1, col])
axes[0, 0].set_ylabel("řez", fontsize=11)
axes[1, 0].set_ylabel("plocha", fontsize=11)
fig.text(0.13, 0.5, "oranžové body: random1 v rozích buněk", fontsize=10, color=GREY)
save(fig, "value_noise.png")

# the blending curves and their slopes, which the smooth ones bring to zero at both ends
t = np.linspace(0, 1, 400)
CURVES = [("lineární  $f$", t, np.ones_like(t), "#222", "-"),
          ("smoothstep  $3f^2 - 2f^3$", smooth(t), 6 * t * (1 - t), ORANGE, "-"),
          ("5. stupeň  $6f^5 - 15f^4 + 10f^3$", t ** 3 * (t * (6 * t - 15) + 10), 30 * t ** 2 * (1 - t) ** 2, "#4C78A8", "--")]
fig, axes = plt.subplots(1, 2, figsize=(9, 2.8), gridspec_kw={"wspace": 0.25})
for label, curve, slope, colour, style in CURVES:
    axes[0].plot(t, curve, style, color=colour, linewidth=2, label=label)
    axes[1].plot(t, slope, style, color=colour, linewidth=2)
axes[1].plot([0, 1], [0, 0], "o", color=ORANGE, markersize=6, clip_on=False, zorder=3)
for ax, title in zip(axes, ("přechod mezi rohy", "směrnice (derivace)")):
    ax.set_title(title, fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 1])
    ax.set_xlabel("$f$", fontsize=11, labelpad=-8)
    ax.tick_params(labelsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_yticks([0, 1])
axes[1].set_yticks([0, 1, 1.5, 1.875])
axes[1].set_yticklabels(["0", "1", "1.5", "1.875"])
fig.legend(*axes[0].get_legend_handles_labels(), fontsize=10, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.0))
save(fig, "smoothstep.png")


# value_noise(p) of task 1 step by step, for one point p in the lesson's own cell (1, 1):
# floor and fract, the four corners, the two mixes along x and the one along y,
# the bend of smoothed, and the same cells mixed straight and bent.
BLUE, GREEN, RED = "#4C78A8", "#54A24B", "#D62728"
STEP_P = (1.7, 1.4)
STEP_C = (1, 1)
STEP_F = (STEP_P[0] - STEP_C[0], STEP_P[1] - STEP_C[1])


def one(cx, cy):
    return float(random1(np.array([cx]), np.array([cy]))[0])


SA, SB = one(STEP_C[0], STEP_C[1]), one(STEP_C[0] + 1, STEP_C[1])
SC, SD = one(STEP_C[0], STEP_C[1] + 1), one(STEP_C[0] + 1, STEP_C[1] + 1)
STEP_TOP, STEP_BOTTOM = mix(SA, SB, STEP_F[0]), mix(SC, SD, STEP_F[0])
STEP_VALUE = mix(STEP_TOP, STEP_BOTTOM, STEP_F[1])


def corner(ax, at, v, label=None, size=360, offset=(0, 0)):
    ax.scatter([at[0]], [at[1]], s=size, c=[[v, v, v]], edgecolors="#222", linewidths=1.3, zorder=5)
    if label:
        ax.annotate(label, at, xytext=offset, textcoords="offset points",
                    ha="center", va="center", fontsize=11, zorder=6)


def blend(weight, x0, y0, w, h, n=160):
    """value_noise over w x h cells from (x0, y0), its f bent by weight."""
    gx, gy = np.meshgrid(x0 + (np.arange(n * w) + 0.5) / n, y0 + (np.arange(n * h) + 0.5) / n)
    cx, cy = np.floor(gx).astype(np.int64), np.floor(gy).astype(np.int64)
    fx, fy = weight(gx - cx), weight(gy - cy)
    a, b = random1(cx, cy), random1(cx + 1, cy)
    c, d = random1(cx, cy + 1), random1(cx + 1, cy + 1)
    return mix(mix(a, b, fx), mix(c, d, fx), fy)


def zoom(ax, title):
    """One cell, (1, 1), with some room around it, y down as in a picture."""
    ax.set_xlim(STEP_C[0] - 0.34, STEP_C[0] + 1.34)
    ax.set_ylim(STEP_C[1] + 1.34, STEP_C[1] - 0.34)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=12, loc="left")
    ax.add_patch(plt.Rectangle(STEP_C, 1, 1, fill=False, ec="#222", lw=1.1))


fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.4), gridspec_kw={"hspace": 0.18, "wspace": 0.2})
(cx, cy), (fx, fy), p = STEP_C, STEP_F, STEP_P

# 1. the cell and the position inside it
ax = axes[0, 0]
ax.add_patch(plt.Rectangle(STEP_C, 1, 1, color="#FDE7CF", zorder=0))
for k in range(4):
    ax.plot([k, k], [0, 3], color="#BBB", lw=0.9)
    ax.plot([0, 3], [k, k], color="#BBB", lw=0.9)
    ax.text(k, -0.15, str(k), ha="center", fontsize=9, color=GREY)
    ax.text(-0.15, k, str(k), va="center", ha="right", fontsize=9, color=GREY)
gx, gy = np.meshgrid(np.arange(4), np.arange(4))
ax.scatter(gx.ravel(), gy.ravel(), s=22, color="#999", zorder=3)
ax.text(1.5, -0.38, "x", ha="center", fontsize=10, color=GREY)
ax.text(-0.42, 1.5, "y", va="center", fontsize=10, color=GREY)
ax.scatter([cx], [cy], s=70, color=ORANGE, zorder=5)
ax.annotate(f"cell = ({cx}, {cy})", (cx, cy), xytext=(-9, -9), textcoords="offset points",
            ha="right", va="top", fontsize=10, color=ORANGE,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
ax.annotate("", (p[0], cy), (cx, cy), arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.8))
ax.annotate("", p, (p[0], cy), arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.8))
ax.text(cx + fx / 2, cy - 0.07, f"f.x = {fx:.1f}", color=BLUE, ha="center", va="bottom", fontsize=10)
ax.text(p[0] + 0.07, cy + fy / 2, f"f.y = {fy:.1f}", color=GREEN, va="center", fontsize=10)
ax.scatter([p[0]], [p[1]], s=80, color=RED, zorder=6)
ax.annotate(f"p = ({p[0]}, {p[1]})", p, xytext=(7, -12), textcoords="offset points", fontsize=10, color=RED)
ax.text(0, 3.22, "Celá část p vybere buňku, desetinná\nčást řekne, kde v ní p leží, od 0 do 1.",
        fontsize=10, color=GREY, va="top")
ax.set_xlim(-0.5, 3.4)
ax.set_ylim(3.75, -0.5)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("1. cell = floor(p), f = fract(p)", fontsize=12, loc="left")

# 2. the four corners
ax = axes[0, 1]
zoom(ax, "2. random1 ve čtyřech rozích")
for (dx, dy), v, name in (((0, 0), SA, "a"), ((1, 0), SB, "b"), ((0, 1), SC, "c"), ((1, 1), SD, "d")):
    corner(ax, (cx + dx, cy + dy), v)
    ax.annotate(f"{name} = {v:.2f}\nrandom1(cell + ({dx}, {dy}))" if dy == 0 else
                f"random1(cell + ({dx}, {dy}))\n{name} = {v:.2f}",
                (cx + dx, cy + dy), xytext=(0, 30 if dy == 0 else -30), textcoords="offset points",
                ha="center", va="center", fontsize=9.5, linespacing=1.4)
ax.text(cx + 0.5, cy + 0.5, "Každý roh mřížky\nmá své pevné číslo,\nsousední buňky\nrohy sdílejí.",
        ha="center", va="center", fontsize=9.5, color=GREY)

# 3. two mixes along x, one along y
ax = axes[0, 2]
zoom(ax, "3. dvakrát mix po x, jednou po y")
ax.imshow(blend(lambda f: f, cx, cy, 1, 1), cmap="gray", vmin=0, vmax=1,
          extent=(cx, cx + 1, cy + 1, cy), zorder=0)
ax.plot([cx, cx + 1], [cy, cy], color=BLUE, lw=2.6, zorder=2)
ax.plot([cx, cx + 1], [cy + 1, cy + 1], color=BLUE, lw=2.6, zorder=2)
ax.plot([p[0], p[0]], [cy, cy + 1], color=GREEN, lw=2.6, zorder=2)
corner(ax, (cx, cy), SA, "a", size=200, offset=(-17, 0))
corner(ax, (cx + 1, cy), SB, "b", size=200, offset=(17, 0))
corner(ax, (cx, cy + 1), SC, "c", size=200, offset=(-17, 0))
corner(ax, (cx + 1, cy + 1), SD, "d", size=200, offset=(17, 0))
corner(ax, (p[0], cy), STEP_TOP, size=170)
corner(ax, (p[0], cy + 1), STEP_BOTTOM, size=170)
ax.text(p[0], cy - 0.09, f"mix(a, b, f.x) = {STEP_TOP:.2f}", color=BLUE, ha="center", va="bottom", fontsize=10)
ax.text(p[0], cy + 1.09, f"mix(c, d, f.x) = {STEP_BOTTOM:.2f}", color=BLUE, ha="center", va="top", fontsize=10)
corner(ax, p, STEP_VALUE, size=260)
ax.scatter([p[0]], [p[1]], s=260, facecolors="none", edgecolors=RED, linewidths=2, zorder=7)
ax.annotate(f"mix({STEP_TOP:.2f}, {STEP_BOTTOM:.2f}, f.y)\n= {STEP_VALUE:.2f}", p, xytext=(-14, 0),
            textcoords="offset points", ha="right", va="center", fontsize=10, color=GREEN,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.88), zorder=8)

# 4. the bend of smoothed
ax = axes[1, 0]
t = np.linspace(0, 1, 200)
ax.plot(t, t, color="#AAA", lw=1.4, ls="--", label="f, bez smoothed")
ax.plot(t, smooth(t), color="#222", lw=2, label="f * f * (3 - 2 f)")
for f, colour, name in ((fx, BLUE, "f.x"), (fy, GREEN, "f.y")):
    s = smooth(f)
    ax.plot([f, f], [0, s], color=colour, lw=1.1, ls=":")
    ax.plot([0, f], [s, s], color=colour, lw=1.1, ls=":")
    ax.scatter([f], [s], color=colour, s=40, zorder=5)
    ax.text(f + 0.02, s - 0.075, f"{name}: {f:.1f} → {s:.2f}", color=colour, fontsize=10)
ax.text(0.03, 0.5, "u kraje buňky\nse f mění pomalu,\nsměrnice je tam 0", fontsize=9, color=GREY)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")
ax.set_xticks([0, 0.5, 1])
ax.set_yticks([0, 0.5, 1])
ax.tick_params(labelsize=9)
ax.set_xlabel("f před ohnutím", fontsize=10)
ax.set_ylabel("f po ohnutí", fontsize=10)
ax.legend(loc="upper left", fontsize=9, frameon=False)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title("4. if smoothed: ohnout f", fontsize=12, loc="left")

# 5 and 6. the same 3 x 2 cells, mixed straight and bent
for ax, weight, title in ((axes[1, 1], lambda f: f, "5. bez smoothed: zlomy na hranách"),
                          (axes[1, 2], smooth, "6. se smoothed: hladce")):
    ax.imshow(blend(weight, 0, 0, 3, 2), cmap="gray", vmin=0, vmax=1, extent=(0, 3, 2, 0))
    for kx in range(4):
        for ky in range(3):
            corner(ax, (kx, ky), one(kx, ky), size=70)
    ax.set_xlim(-0.1, 3.1)
    ax.set_ylim(2.1, -0.1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=12, loc="left")
save(fig, "value_noise_steps.png")


# 2. fBm: four octaves and their sum --------------------------------------------
H, W, SCALE = 160, 256, 96.0
x = (np.arange(W)[None, :] + 0.5) / SCALE
y = (np.arange(H)[:, None] + 0.5) / SCALE
fig, axes = plt.subplots(1, 5, figsize=(13, 2.2), gridspec_kw={"wspace": 0.08})
ox, oy = x, y
for k in range(4):
    axes[k].imshow(value_noise(ox, oy), cmap="gray", vmin=0, vmax=1)
    weight = "1" if k == 0 else f"1/{2 ** k}"
    axes[k].set_title(f"oktáva {k + 1}, váha {weight}", fontsize=11)
    ox, oy = ox * 2.0 + 0.5, oy * 2.0 + 0.5
    bare(axes[k])
axes[4].imshow(fbm(x, y, 4), cmap="gray", vmin=0, vmax=1)
axes[4].set_title("součet / součet vah", fontsize=11)
bare(axes[4])
for k in range(4):
    axes[k].text(1.04, 0.5, "+" if k < 3 else "=", transform=axes[k].transAxes, fontsize=16, ha="center", va="center")
save(fig, "fbm.png")

# Why fBm moves each octave by half a cell: one row of five octaves at lacunarity 2
# around x = 1, a grid line of octave 1. Smoothstep is flat along x on every cell
# edge. Without the shift x = 1 is an edge of all five octaves, so their sum is flat
# there too, with the shift it is an edge of octave 1 only.
SHIFT_ROW, SHIFT_EPS = 0.37, 1e-6


def shifted_octave(x, k, shift):
    """Octave k, 1 the coarsest, along the row SHIFT_ROW, x in cells of octave 1."""
    y = np.full_like(x, SHIFT_ROW)
    for _ in range(k - 1):
        x, y = x * 2.0 + shift, y * 2.0 + shift
    return value_noise(x, y)


def shifted_sum(x, shift):
    return sum(0.5 ** (k - 1) * shifted_octave(x, k, shift) for k in range(1, 6)) / (2 - 0.5 ** 4)


def tangent_at_one(ax, f, half):
    """Draws the tangent of f at x = 1 and gives its slope."""
    at = np.array([1.0])                                 # arrays, as the hash wants them
    h = float(f(at)[0])
    s = float((f(at + SHIFT_EPS) - f(at - SHIFT_EPS))[0] / (2 * SHIFT_EPS))
    ax.plot([1 - half, 1 + half], [h - s * half, h + s * half], color="black", lw=2.2, zorder=5)
    ax.plot([1], [h], "o", color="black", ms=4, zorder=6)
    return s


xs = np.linspace(0.5, 1.5, 1500)
fig, axes = plt.subplots(6, 2, figsize=(12, 9.5), sharex=True,
                         gridspec_kw={"height_ratios": [1, 1, 1, 1, 1, 1.3], "hspace": 0.35, "wspace": 0.08})
for col, shift, colour, title in ((0, 0.0, RED, "bez posunu,  p = p * 2"),
                                  (1, 0.5, BLUE, "s posunem,  p = p * 2 + 0.5")):
    for k in range(1, 6):
        ax = axes[k - 1, col]
        f = lambda x, k=k: shifted_octave(x, k, shift)
        ax.plot(xs, f(xs), color=colour, lw=1.2)
        size, offset = 2 ** (k - 1), (2 ** (k - 1) - 1) * shift    # octave k is at x * size + offset
        edges = np.arange(np.ceil(0.5 * size + offset), np.floor(1.5 * size + offset) + 1)
        for edge in (edges - offset) / size:
            ax.axvline(edge, color="#CCC", lw=0.7, zorder=0)
        s = tangent_at_one(ax, f, 0.035)
        flat = abs(s) < 0.02
        ax.text(1.0, 1.08, "plochá" if flat else f"sklon {s:+.1f}", transform=ax.get_xaxis_transform(),
                ha="center", fontsize=9, fontweight="bold" if flat else "normal")
        ax.set_ylim(-0.05, 1.05)
        if col == 0:
            ax.set_ylabel(f"oktáva {k}\nváha {0.5 ** (k - 1):g}", fontsize=9, rotation=0, ha="right", va="center")
    ax = axes[5, col]
    f = lambda x: shifted_sum(x, shift)
    ax.plot(xs, f(xs), color=colour, lw=2)
    s = tangent_at_one(ax, f, 0.05)
    ax.text(1.0, 1.04, "součet plochý" if abs(s) < 0.02 else f"součet má sklon {s:+.2f}",
            transform=ax.get_xaxis_transform(), ha="center", fontsize=10, fontweight="bold")
    if col == 0:
        ax.set_ylabel("součet", fontsize=10, rotation=0, ha="right", va="center")
    ax.set_xticks([0.5, 0.75, 1, 1.25, 1.5])
    ax.set_xlabel("x v buňkách první oktávy", fontsize=10)
    axes[0, col].set_title(title + "\n", fontsize=11, color=colour)
    for row in axes[:, col]:
        row.axvspan(0.99, 1.01, color="#FFE9A8", zorder=-1)
        row.set_yticks([])
        for side in ("top", "right", "left"):
            row.spines[side].set_visible(False)
fig.text(0.5, 0.015, "šedé čáry jsou hrany buněk té oktávy, na nich je smoothstep po x plochý",
         ha="center", fontsize=10, color=GREY)
save(fig, "fbm_shift.png")


# 3. Voronoi step by step: the grid, the jittered points, one pixel's 3 x 3 cells,
# its F1 and F2, then F1 and F2 - F1 of every pixel. The cells are the kernel's own,
# at jitter 1, from cell ORIGIN on, picked so that the pixel's nearest point lies
# in the next cell, which is why the kernel looks at the neighbours.
GW, GH = 5, 4
ORIGIN = np.array([59, 9])
HOME = np.array([2, 1])                                    # the pixel's cell in the picture
PIXEL = HOME + np.array([0.85, 0.35])
cells_x, cells_y = np.meshgrid(np.arange(-2, GW + 2), np.arange(-2, GH + 2))
jx, jy = random2((cells_x + ORIGIN[0]).ravel(), (cells_y + ORIGIN[1]).ravel())
cells = np.stack([cells_x.ravel(), cells_y.ravel()], axis=-1)
points = cells + np.stack([jx, jy], axis=-1)               # c + 0.5 + (random2(c) - 0.5) * 1
centres = cells + 0.5
shown = (cells[:, 0] >= 0) & (cells[:, 0] < GW) & (cells[:, 1] >= 0) & (cells[:, 1] < GH)
near = np.all(np.abs(cells - HOME) <= 1, axis=-1)
R = 100                                                    # raster pixels a cell
ry, rx = np.mgrid[0:GH * R, 0:GW * R]
at = np.stack([(rx + 0.5) / R, (ry + 0.5) / R], axis=-1)
dist = np.linalg.norm(at[:, :, None, :] - points[None, None], axis=-1)
order = np.sort(dist, axis=-1)
f1, f2 = order[..., 0], order[..., 1]
owner = cells[np.argmin(dist, axis=-1)] + ORIGIN           # the cell of the nearest point
ox, oy = owner[..., 0], owner[..., 1]
r2x, r2y = random2(ox.ravel(), oy.ravel())
r1 = random1((ox - 31).ravel(), (oy + 17).ravel())
cell_colour = np.stack([r2x, r2y, r1], axis=-1).reshape(GH * R, GW * R, 3)   # the kernel's Cell
pale = 0.35 * cell_colour + 0.65


def panel(ax, title, background=None, grey=None):
    if background is not None:
        ax.imshow(background, extent=(0, GW, GH, 0))
    if grey is not None:
        ax.imshow(np.clip(grey, 0, 1), cmap="gray", vmin=0, vmax=1, extent=(0, GW, GH, 0))
    for k in range(GW + 1):
        ax.axvline(k, color="#AAA", linewidth=0.8, linestyle=":")
    for k in range(GH + 1):
        ax.axhline(k, color="#AAA", linewidth=0.8, linestyle=":")
    ax.set_xlim(0, GW)
    ax.set_ylim(GH, 0)
    ax.set_aspect("equal")
    bare(ax)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color("#CCC")
    ax.set_title(title, fontsize=12)


def dots(ax, which, colour="#222", size=5):
    ax.plot(points[which, 0], points[which, 1], "o", color=colour, markersize=size)


fig, axes = plt.subplots(2, 3, figsize=(12, 7.1), gridspec_kw={"wspace": 0.06, "hspace": 0.16})
axes = axes.ravel()

panel(axes[0], "1. bod ve středu každé buňky")
axes[0].plot(centres[shown, 0], centres[shown, 1], "o", color="#222", markersize=5)

panel(axes[1], "2. posun o (random2 − 0.5) · jitter")
axes[1].plot(centres[shown, 0], centres[shown, 1], "o", markerfacecolor="none", markeredgecolor="#999", markersize=5)
for c, q in zip(centres[shown], points[shown]):
    axes[1].annotate("", q, c, arrowprops={"arrowstyle": "-|>", "color": GREY, "linewidth": 1, "shrinkA": 3, "shrinkB": 3})
dots(axes[1], shown)

ax = axes[2]
panel(ax, "3. pixel měří k bodům 3 × 3 buněk")
ax.add_patch(plt.Rectangle(HOME - 1, 3, 3, color=ORANGE, alpha=0.12, linewidth=0))
ax.add_patch(plt.Rectangle(HOME, 1, 1, fill=False, edgecolor=ORANGE, linewidth=1.6))
for q in points[near]:
    ax.plot([PIXEL[0], q[0]], [PIXEL[1], q[1]], color=GREY, linewidth=0.8)
dots(ax, shown & ~near, colour="#BBB")
dots(ax, near)
ax.plot(*PIXEL, "s", color=ORANGE, markersize=8)

ax = axes[3]
panel(ax, "4. F1, F2 a barva buňky (Cell)", background=pale)
d = np.linalg.norm(points - PIXEL, axis=-1)
first, second = points[np.argsort(d)[:2]]
for target, colour, label in ((first, ORANGE, "F1"), (second, GREY, "F2")):
    ax.annotate("", target, PIXEL, arrowprops={"arrowstyle": "-|>", "color": colour, "linewidth": 2, "shrinkA": 6, "shrinkB": 4})
    along = (target - PIXEL) / np.linalg.norm(target - PIXEL)
    side = np.array([along[1], -along[0]])                # perpendicular to the arrow
    if (side[1] > 0) != (label == "F1"):                  # F1 below its arrow, F2 above
        side = -side
    spot = (PIXEL + target) / 2 + 0.22 * side
    ax.text(spot[0], spot[1], label, color=colour, fontsize=13, ha="center", va="center", weight="bold")
dots(ax, shown)
ax.plot(*PIXEL, "s", color=ORANGE, markersize=8)

panel(axes[4], "5. F1 v každém pixelu", grey=f1)
dots(axes[4], shown, colour=ORANGE, size=3)
panel(axes[5], "6. F2 - F1 v každém pixelu", grey=f2 - f1)
save(fig, "voronoi.png")


def voronoi(px, py, jitter=1.0):
    """F1, F2 and the nearest point's cell at positions px, py in cells, as the kernel finds them."""
    hx, hy = np.floor(px).astype(np.int64), np.floor(py).astype(np.int64)
    f1 = np.full(px.shape, 8.0)
    f2 = np.full(px.shape, 8.0)
    ox, oy = hx.copy(), hy.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            cx, cy = hx + dx, hy + dy
            jx, jy = random2(cx, cy)
            d = np.hypot(px - (cx + 0.5 + (jx - 0.5) * jitter), py - (cy + 0.5 + (jy - 0.5) * jitter))
            nearer = d < f1
            f2 = np.where(nearer, f1, np.minimum(f2, d))
            f1 = np.where(nearer, d, f1)
            ox, oy = np.where(nearer, cx, ox), np.where(nearer, cy, oy)
    return f1, f2, ox, oy


def colours(*hexes):
    return [np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]) for h in hexes]


def ramp(t, a, b):
    t = np.clip(t, 0, 1)[..., None]
    return a + (b - a) * t


# Voronoi in nature: four photos and the lesson's Voronoi noise made to look like each.
# The photos come from Wikimedia Commons, as imgs/2/SOURCES.md says. Jitter stays below 1,
# because two points of neighbouring cells that nearly meet keep F2 - F1 small all around
# them and draw a pale wedge, and each panel starts at a cell picked for points well apart.
N = 240
ny, nx = np.mgrid[0:N, 0:N]
NATURE = []

f1, f2, ox, oy = voronoi((nx + 0.5) / 62 + 9, (ny + 0.5) / 62 + 30, jitter=0.85)      # giraffe: Cell darker inside, cream edges
dark, brown, cream = colours("#3B2414", "#8A5A33", "#E4D3B4")
patch = ramp(f1 * 1.3 + 0.3 * random1(ox, oy) - 0.1, dark, brown)
NATURE.append(("giraffe.jpg", "flowcomm, CC BY 2.0", "žirafa\nCell, světlý okraj z F2 - F1",
               ramp(((f2 - f1) - 0.05) / 0.06, cream, patch)))

f1, f2, ox, oy = voronoi((nx + 0.5) / 34 + 6, (ny + 0.5) / 34 + 26, jitter=0.85)     # dragonfly wing: thin dark F2 - F1 lines
pale, vein = colours("#C9C6B4", "#2B2A22")
NATURE.append(("dragonfly_wing.jpg", "Patilssp, CC BY 4.0", "křídlo vážky\nF2 - F1 jako tenké žilky",
               ramp(((f2 - f1) - 0.015) / 0.03, vein, pale)))

f1, f2, ox, oy = voronoi((nx + 0.5) / 70 + 48, (ny + 0.5) / 70 + 6, jitter=0.85)     # dried mud: wide dark cracks
mud, crack = colours("#7E776C", "#1E1A16")
plate = mud * (0.85 + 0.3 * random1(ox, oy)[..., None])
NATURE.append(("dried_mud.jpg", "Novoklimov, CC0", "vyschlé bahno\nF2 - F1 jako praskliny",
               ramp(((f2 - f1) - 0.05) / 0.05, crack, plate)))

f1, f2, ox, oy = voronoi((nx + 0.5) / 48 + 39, (ny + 0.5) / 48 + 39, jitter=0.6)   # basalt: Cell, less jitter
rock, gap = colours("#A7A195", "#2C2A27")
top = rock * (0.72 + 0.45 * random1(ox, oy)[..., None])
NATURE.append(("basalt.jpg", "David Singleton, CC BY 2.0", "čedičové sloupy\nCell s jitter 0.6",
               ramp(((f2 - f1) - 0.04) / 0.05, gap, top)))

fig, axes = plt.subplots(2, 4, figsize=(12, 6.6), gridspec_kw={"hspace": 0.16, "wspace": 0.06})
for col, (photo, credit, title, imitation) in enumerate(NATURE):
    axes[0, col].imshow(plt.imread(os.path.join(OUT, photo)))
    axes[0, col].set_title(credit, fontsize=8, color=GREY, y=-0.09)
    axes[1, col].imshow(np.clip(imitation, 0, 1), interpolation="bilinear")
    axes[1, col].set_title(title, fontsize=10, y=-0.2)
    bare(axes[0, col])
    bare(axes[1, col])
fig.savefig(os.path.join(OUT, "voronoi_nature.jpg"), dpi=150, bbox_inches="tight", facecolor="white",
            pil_kwargs={"quality": 88})                    # photos, so JPEG
plt.close(fig)


# Fractal Voronoi, picture only: octaves of F1 and of F2 - F1, each twice as fine
# and half as strong, as fBm sums value noise.
def fractal(px, py, octaves, edges):
    total, weight, amplitude = 0.0, 0.0, 1.0
    for _ in range(octaves):
        f1, f2, _, _ = voronoi(px, py)
        total = total + amplitude * (f2 - f1 if edges else f1)
        weight += amplitude
        amplitude *= 0.5
        px, py = px * 2.0 + 0.5, py * 2.0 + 0.5
    return total / weight


FH, FW = 200, 300
fy, fx = np.mgrid[0:FH, 0:FW]
fpx, fpy = (fx + 0.5) / 90 + 4, (fy + 0.5) / 90 + 4
FRACTAL = [("F1, 1 oktáva", fractal(fpx, fpy, 1, False)),
           ("F1, 5 oktáv", fractal(fpx, fpy, 5, False)),
           ("F2 - F1, 1 oktáva", fractal(fpx, fpy, 1, True)),
           ("F2 - F1, 5 oktáv", fractal(fpx, fpy, 5, True))]
fig, axes = plt.subplots(1, 4, figsize=(13, 2.6), gridspec_kw={"wspace": 0.06})
for ax, (title, img) in zip(axes, FRACTAL):
    ax.imshow(img, cmap="gray", vmin=0, vmax=np.percentile(img, 99.5))
    ax.set_title(title, fontsize=11)
    bare(ax)
save(fig, "voronoi_fractal.png")


# 4. The terrain's colour stops ----------------------------------------------------
t = np.linspace(-0.26, 0.3, 1200)
lab = np.stack([np.interp(t, STOP_AT, STOP_LAB[:, k]) for k in range(3)], axis=-1)
strip = np.clip(linear_to_srgb(lab_to_linear(lab)), 0, 1)
fig = plt.figure(figsize=(12, 2.6))
ax = fig.add_axes([0.03, 0.5, 0.94, 0.34])
ax.imshow(np.repeat(strip[None], 30, axis=0), aspect="auto", extent=(t[0], t[-1], 0, 1))
ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)
ax.set_xticks([])
# the stops by the sea lie too close for their labels, which are spread apart
# and joined to their stops by a bent line
GAP = 0.05
spot = np.array([at for at, *_ in nb.STOPS], dtype=np.float64)
for _ in range(500):
    for k in range(len(spot) - 1):
        short = GAP - (spot[k + 1] - spot[k])
        if short > 0:
            spot[k] -= short / 2
            spot[k + 1] += short / 2
for (at, _, label, _), x_label in zip(nb.STOPS, spot):
    ax.plot([at, at, x_label], [0, -0.12, -0.42], color="#333", linewidth=0.9, clip_on=False)
    ax.text(x_label, -0.48, f"{label}\n{at:+.3f}".replace("+0.000", "0"), ha="center", va="top", fontsize=9, clip_on=False)
ax.axvline(0, color="white", linewidth=1.5)
ax.text(0, 1.08, "hladina moře, h - sea = 0", ha="center", va="bottom", fontsize=11)
ax.text(t[0], 1.08, "h - sea", ha="left", va="bottom", fontsize=11, color=GREY)
save(fig, "terrain_stops.png")


# 5. A section through the terrain: normals, the sun, and the brightness from above ---
xs = np.linspace(0, 4.0, 2000)
height = fbm(xs * 1.2 + 0.3, np.full_like(xs, 2.3), 4)
sea = 0.5
RELIEF = 1.6                                               # how much taller the drawing is than the noise
top = (np.maximum(height, sea) - sea) * RELIEF             # the drawn ground, 0 at the sea
depth = (height - sea) * RELIEF                            # the bottom under the water
slope = np.gradient(top, xs)
normal = np.stack([-slope, np.ones_like(slope)], axis=-1)
normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
sun = np.array([-1.0, 1.0]) / np.sqrt(2.0)                 # l, towards the upper left, 45 degrees up
bright = 0.25 + 0.75 * np.maximum(normal @ sun, 0.0)
under = height < sea
floor = min(depth.min(), 0.0) - 0.08
peak = top.max()
LAND, WATER = np.array([0.44, 0.62, 0.25]), np.array([0.17, 0.42, 0.64])

fig = plt.figure(figsize=(10, 3.2))
ax = fig.add_axes([0.0, 0.24, 1.0, 0.76])
ax.fill_between(xs, floor, top, where=~under, color=LAND, linewidth=0, interpolate=True)
ax.fill_between(xs, floor, top, where=under, color=WATER, linewidth=0, interpolate=True)
ax.plot(xs, np.where(under, depth, np.nan), color="white", linewidth=1, linestyle=":")
ax.plot(xs, top, color="#222", linewidth=1.4)


def arrow(start, direction, colour, length=0.13):
    ax.annotate("", start + length * direction, start, arrowprops={"arrowstyle": "-|>", "color": colour, "linewidth": 1.6, "shrinkA": 0, "shrinkB": 0})


# n and l side by side on the tallest hill's sunny slope, where they still part
summit = int(np.argmax(top))
left = summit
while left > 0 and not under[left - 1]:
    left -= 1
hill = np.full(len(xs), np.inf)
hill[left:summit] = np.abs(normal[left:summit] @ sun - 0.85)
facing = int(np.argmin(hill))
foot = np.array([xs[facing], top[facing]])
for k in range(50, len(xs), 110):
    if abs(xs[k] - foot[0]) > 0.1:
        arrow(np.array([xs[k], top[k]]), normal[k], ORANGE)
arrow(foot, normal[facing], ORANGE, 0.2)
arrow(foot, sun, "#C9A227", 0.2)
ax.text(*(foot + 0.25 * normal[facing]), "n", color=ORANGE, fontsize=13, weight="bold", ha="center", va="center")
ax.text(*(foot + 0.25 * sun), "l", color="#C9A227", fontsize=13, weight="bold", ha="center", va="center")
sun_at = np.array([0.12, peak + 0.22])
ax.plot(*sun_at, "o", color="#F2C14E", markersize=20)
ax.text(sun_at[0] + 0.12, sun_at[1], "slunce", fontsize=11, va="center")
# the widest stretch of water carries the note about the flat surface
runs, start = [], None
for k, wet in enumerate(np.append(under, False)):
    if wet and start is None:
        start = k
    elif not wet and start is not None:
        runs.append((k - start, start, k))
        start = None
_, a, b = max(runs)
middle = min(max((xs[a] + xs[b - 1]) / 2, 0.4), xs[-1] - 0.4)
ax.text(middle, -0.04, "hladina je rovná", ha="center", va="top", fontsize=10, color="white")
ax.set_title("Lambert: jas = 0.25 + 0.75 · max(n · l, 0),  n normála,  l ke slunci", fontsize=11, color="#333")
ax.set_xlim(0, xs[-1])
ax.set_ylim(floor, peak + 0.34)
ax.set_aspect("equal", anchor="S")
bare(ax)
# what a kernel computes for these pixels, seen from above
strip_ax = fig.add_axes([0.0, 0.02, 1.0, 0.09])
colour = linear_to_srgb(srgb_to_linear(np.where(under[:, None], WATER, LAND)) * bright[:, None])
strip_ax.imshow(colour[None], aspect="auto", extent=(0, xs[-1], 0, 1))
strip_ax.set_xlim(0, xs[-1])
bare(strip_ax)
strip_ax.set_title("pohled shora: barva krát jas", fontsize=10, color=GREY)
fig.canvas.draw()
left, right = ax.get_position().x0, ax.get_position().x1     # the strip as wide as the section
strip_ax.set_position([left, 0.02, right - left, 0.09])
save(fig, "normal.png")



# light(at) of task 5 step by step, for one pixel on a slope of the goal's terrain:
# the four neighbours' heights, the central differences, the normal, the sun, Lambert,
# and the brightness of the land around the pixel.
L_SEA, L_RELIEF, L_SUN = 0.5, 400.0, 135.0
L_AT = (910, 381)                                          # x, y of a slope facing the sun
L_H = np.maximum(_goal, L_SEA)                              # the water counts as flat, at sea
L_DX = np.zeros_like(L_H)
L_DY = np.zeros_like(L_H)
L_DX[:, 1:-1] = (L_H[:, 2:] - L_H[:, :-2]) * 0.5
L_DY[1:-1, :] = (L_H[2:, :] - L_H[:-2, :]) * 0.5
L_N = np.dstack([-L_RELIEF * L_DX, -L_RELIEF * L_DY, np.ones_like(L_H)])
L_N /= np.linalg.norm(L_N, axis=2, keepdims=True)
L_L = np.array([np.cos(np.radians(L_SUN)), -np.sin(np.radians(L_SUN)), 1.0])
L_L /= np.linalg.norm(L_L)
L_BRIGHT = 0.25 + 0.75 * np.maximum(L_N @ L_L, 0.0)
lx, ly = L_AT
h_left, h_right = L_H[ly, lx - 1], L_H[ly, lx + 1]
h_up, h_down = L_H[ly - 1, lx], L_H[ly + 1, lx]
dx, dy = L_DX[ly, lx], L_DY[ly, lx]
n_at = L_N[ly, lx]
nl_at = float(n_at @ L_L)

fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.4), gridspec_kw={"hspace": 0.32, "wspace": 0.28})

# 1. the heights of the four neighbours
ax = axes[0, 0]
R = 3
patch = L_H[ly - R:ly + R + 1, lx - R:lx + R + 1]
ax.imshow(patch, cmap="gray", extent=(-R - 0.5, R + 0.5, R + 0.5, -R - 0.5))
for k in range(-R, R + 2):
    ax.plot([k - 0.5, k - 0.5], [-R - 0.5, R + 0.5], color="white", lw=0.6, alpha=0.5)
    ax.plot([-R - 0.5, R + 0.5], [k - 0.5, k - 0.5], color="white", lw=0.6, alpha=0.5)
ax.add_patch(plt.Rectangle((-0.5, -0.5), 1, 1, fill=False, ec=ORANGE, lw=2.4))
for (ox, oy), v, colour in (((1, 0), h_right, BLUE), ((-1, 0), h_left, BLUE), ((0, -1), h_up, GREEN), ((0, 1), h_down, GREEN)):
    ax.add_patch(plt.Rectangle((ox - 0.5, oy - 0.5), 1, 1, fill=False, ec=colour, lw=2.2))
    ax.text(ox * 1.95, oy * 1.95, f"{v:.4f}", ha="center", va="center", fontsize=9.5, color=colour,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))
ax.set_xlim(-R - 0.5, R + 0.5)
ax.set_ylim(R + 0.5, -R - 0.5)
bare(ax)
ax.set_title("1. height(at) čtyř sousedů", fontsize=12, loc="left")
ax.text(-R - 0.5, R + 1.0, f"dx = ({h_right:.4f} - {h_left:.4f}) / 2 = {dx:.5f}\n"
        f"dy = ({h_down:.4f} - {h_up:.4f}) / 2 = {dy:.5f}", fontsize=9.5, va="top", family="monospace")

# 2. the slope along the row, a central difference
ax = axes[0, 1]
offsets = np.arange(-3, 4)
row = L_H[ly, lx - 3:lx + 4]
ax.plot(offsets, row, "o-", color="#999", lw=1, ms=5)
ax.plot([-1, 1], [h_left, h_right], "o", color=BLUE, ms=8, zorder=4)
ax.plot([0], [L_H[ly, lx]], "s", color=ORANGE, ms=9, zorder=4)
ax.plot([-2.2, 2.2], [L_H[ly, lx] - 2.2 * dx, L_H[ly, lx] + 2.2 * dx], color=BLUE, lw=1.6, ls="--")
ax.annotate("", (1, h_right), (1, h_left), arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.2))
ax.text(1.12, (h_left + h_right) / 2, "vpravo - vlevo", color=BLUE, fontsize=9, va="center")
ax.annotate("", (1, h_left), (-1, h_left), arrowprops=dict(arrowstyle="<->", color=GREY, lw=1.2))
ax.text(0, h_left, "2 pixely", color=GREY, fontsize=9, ha="center", va="top")
ax.set_xticks(offsets)
ax.set_xticklabels([f"{k:+d}" if k else "at" for k in offsets], fontsize=9)
ax.tick_params(axis="y", labelsize=8)
ax.set_xlabel("pixely v řádku", fontsize=10)
ax.set_ylabel("výška", fontsize=10)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title("2. sklon z rozdílu sousedů", fontsize=12, loc="left")
ax.text(0.02, 0.98, "(dx, dy) je gradient výšky,\nodkud se bere, odvodíme\nv lekci 3 u hran",
        transform=ax.transAxes, fontsize=9.5, va="top", color="#333",
        bbox=dict(boxstyle="round,pad=0.35", fc="#FDE7CF", ec="none"))

# 3. the normal, perpendicular to the steepened slope
ax = axes[0, 2]
k = L_RELIEF * dx
t = np.array([1.0, k]) / np.hypot(1.0, k)
n2 = np.array([-k, 1.0]) / np.hypot(1.0, k)
ax.fill_between([-1.4, 1.4], [-1.4 * k - 1, 1.4 * k - 1], [-1.4 * k, 1.4 * k], color="#70A040", alpha=0.35, lw=0)
ax.plot([-1.4, 1.4], [-1.4 * k, 1.4 * k], color="#222", lw=1.6)
ax.plot([-1.4, 1.4], [-1.4 * dx, 1.4 * dx], color=GREY, lw=1, ls=":")
ax.text(1.42, 1.4 * dx, "bez převýšení", fontsize=9, color=GREY, va="center")
for vec, colour, name, place in ((t, BLUE, "svah (1, r·dx)", (0.15, -0.16)), (n2, ORANGE, "n (-r·dx, 1)", (-0.05, 0.1))):
    ax.annotate("", vec, (0, 0), arrowprops=dict(arrowstyle="-|>", color=colour, lw=2))
    ax.text(vec[0] + place[0], vec[1] + place[1], name, color=colour, fontsize=10,
            ha="left" if vec[0] > 0 else "right", va="center")
c = 0.13
ax.plot([c * t[0], c * (t[0] + n2[0]), c * n2[0]], [c * t[1], c * (t[1] + n2[1]), c * n2[1]], color="#222", lw=1)
ax.text(-1.35, -1.25, f"r = relief = {L_RELIEF:.0f}\nr·dx = {k:.2f}\nsvah · n = 0, n je kolmá", fontsize=9.5, va="bottom")
ax.set_xlim(-1.5, 2.1)
ax.set_ylim(-1.3, 1.3)
ax.set_aspect("equal")
bare(ax)
ax.set_title("3. normála v řezu x, z", fontsize=12, loc="left")

# 4. the direction to the sun, seen from above
ax = axes[1, 0]
sx, sy = np.cos(np.radians(L_SUN)), -np.sin(np.radians(L_SUN))
ax.add_patch(plt.Circle((0, 0), 1, fill=False, ec="#CCC", lw=1))
ax.annotate("", (1.3, 0), (0, 0), arrowprops=dict(arrowstyle="-|>", color="#222", lw=1.2))
ax.annotate("", (0, 1.3), (0, 0), arrowprops=dict(arrowstyle="-|>", color="#222", lw=1.2))
ax.text(1.33, 0, "x", va="center", fontsize=11)
ax.text(0, 1.36, "y", ha="center", va="top", fontsize=11)
arc = np.radians(np.linspace(0, L_SUN, 60))
ax.plot(0.35 * np.cos(arc), -0.35 * np.sin(arc), color=GREY, lw=1.2)
ax.text(0.4, -0.3, f"s = {L_SUN:.0f}°", fontsize=10, color=GREY)
ax.annotate("", (sx, sy), (0, 0), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.2))
ax.add_patch(plt.Circle((sx * 1.18, sy * 1.18), 0.09, color="#F5C04F"))
ax.text(sx * 1.18, sy * 1.18 - 0.17, "slunce", ha="center", fontsize=10)
ax.text(1.6, -0.95, "(cos s, -sin s):\ny míří dolů,\nproto -sin.\n\nTřetí složka 1\nzvedne slunce\n45° nad obzor.",
        fontsize=9.5, va="top")
ax.set_xlim(-1.45, 2.95)
ax.set_ylim(1.45, -1.45)
ax.set_aspect("equal")
bare(ax)
ax.set_title("4. směr ke slunci, pohled shora", fontsize=12, loc="left")

# 5. Lambert: brightness from n · l
ax = axes[1, 1]
c = np.linspace(-1, 1, 400)
ax.plot(c, 0.25 + 0.75 * np.maximum(c, 0), color="#222", lw=2)
ax.axhline(0.25, color=GREY, lw=0.8, ls=":")
flat = float(np.array([0, 0, 1.0]) @ L_L)
for value, label, colour, dy_text in ((nl_at, "náš pixel", ORANGE, 0.04), (flat, "rovina a moře", BLUE, 0.03)):
    b = 0.25 + 0.75 * value
    ax.plot([value], [b], "o", color=colour, ms=7, zorder=4)
    ax.text(value - 0.03, b + dy_text, f"{label}: n · l = {value:.2f}, jas {b:.2f}", color=colour,
            fontsize=9, ha="right")
ax.text(-0.95, 0.29, "odvrácený svah: 0.25", fontsize=9, color=GREY)
ax.set_xlim(-1, 1)
ax.set_ylim(0, 1.08)
ax.set_xlabel("n · l, kosinus úhlu mezi n a l", fontsize=10)
ax.set_ylabel("jas", fontsize=10)
ax.tick_params(labelsize=9)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title("5. 0.25 + 0.75 max(n · l, 0)", fontsize=12, loc="left")

# 6. the brightness of the land around the pixel
ax = axes[1, 2]
W2, H2 = 180, 130
crop = L_BRIGHT[ly - H2 // 2:ly + H2 // 2, lx - W2 // 2:lx + W2 // 2]
ax.imshow(crop, cmap="gray", vmin=0.2, vmax=1, interpolation="nearest")
ax.add_patch(plt.Rectangle((W2 // 2 - 3.5, H2 // 2 - 3.5), 7, 7, fill=False, ec=ORANGE, lw=2))
ax.annotate("", (W2 - 40 + 22 * sx, 24 + 22 * sy), (W2 - 40, 24),
            arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.2))
ax.text(W2 - 40, 30, "ke slunci", color=ORANGE, fontsize=9, ha="center", va="top",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
bare(ax)
ax.set_title("6. light(at) v okolí pixelu", fontsize=12, loc="left")
save(fig, "light_steps.png")

print("written to", OUT)
