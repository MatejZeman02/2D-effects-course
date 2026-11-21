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

print("written to", OUT)
