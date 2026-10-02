"""Draws the lesson 0 diagrams and its test picture into imgs/0 of the lecture folder."""
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

OUT = str(Path(__file__).resolve().parent.parent / "imgs" / "0")
os.makedirs(OUT, exist_ok=True)

BLUE = "#4C78A8"
ORANGE = "#F58518"
PURPLE = "#8E6BBF"
GREEN = "#54A24B"
GREY = "#6B6B6B"
LIGHT = {"blue": "#DCE7F3", "orange": "#FDE6CF", "purple": "#E9E1F4", "green": "#DDEFD9", "grey": "#EEEEEE"}


def box(ax, x, y, w, h, title, lines, edge, fill, title_size=13, line_size=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                linewidth=2, edgecolor=edge, facecolor=fill))
    ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=title_size, weight="bold", color="#222")
    for i, line in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.62 - i * 0.30, line, ha="center", va="top", fontsize=line_size, color="#333")


def arrow(ax, x0, y0, x1, y1, color=GREY):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=18, linewidth=2, color=color))


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# 1. The course loop -------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 1.8))
ax.set_xlim(0, 11)
ax.set_ylim(0, 1.75)
ax.axis("off")
box(ax, 0.1, 0.1, 3.1, 1.55, "1. Kernel v gmacs", ["Sára + Jupyter", "jeden pixel, jedno vlákno", "posuvníky v reálném čase"], BLUE, LIGHT["blue"])
box(ax, 3.95, 0.1, 3.1, 1.55, "2. Funkce v NumPy", ["Jupyter", "stejná matematika", "na celém poli najednou"], ORANGE, LIGHT["orange"])
box(ax, 7.8, 0.1, 3.1, 1.55, "3. Plugin v Kritě", ["Krita 6", "funkce beze změny", "základ semestrální práce"], PURPLE, LIGHT["purple"])
arrow(ax, 3.25, 0.88, 3.9, 0.88)
arrow(ax, 7.1, 0.88, 7.75, 0.88)
save(fig, "course_loop.png")


# 2. An image is an array ----------------------------------------------------
W, H = 8, 6
yy, xx = np.mgrid[0:H, 0:W]
img = np.zeros((H, W, 3))
img[..., 0] = 0.15 + 0.8 * xx / (W - 1)
img[..., 1] = 0.25 + 0.5 * (1 - yy / (H - 1))
img[..., 2] = 0.85 - 0.7 * xx / (W - 1)
PX, PY = 5, 2
value = np.append(img[PY, PX], 1.0)

fig = plt.figure(figsize=(11, 4.6))
ax = fig.add_axes([0.03, 0.12, 0.5, 0.78])
ax.imshow(img, interpolation="nearest", extent=(-0.5, W - 0.5, H - 0.5, -0.5))
for i in range(W + 1):
    ax.axvline(i - 0.5, color="white", linewidth=1)
for j in range(H + 1):
    ax.axhline(j - 0.5, color="white", linewidth=1)
ax.add_patch(Rectangle((PX - 0.5, PY - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=3.5))
ax.set_xticks(range(W))
ax.set_yticks(range(H))
ax.xaxis.tick_top()
ax.set_xlabel("x (sloupec)  →", fontsize=12)
ax.xaxis.set_label_position("top")
ax.set_ylabel("↓  y (řádek)", fontsize=12)
ax.tick_params(length=0, labelsize=10)
for s in ax.spines.values():
    s.set_visible(False)
ax.text(-0.5, H + 0.15, "počátek (0, 0) je vlevo nahoře", fontsize=10, color=GREY, va="top")

bx = fig.add_axes([0.6, 0.30, 0.36, 0.52])
names = ["R", "G", "B", "A"]
colors = ["#D9534F", "#5CB85C", "#428BCA", "#999999"]
bx.bar(names, value, color=colors, width=0.6)
for i, v in enumerate(value):
    bx.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=11)
bx.set_ylim(0, 1.2)
bx.set_yticks([0, 0.5, 1.0])
bx.set_title(f"pixel x = {PX}, y = {PY}: čtyři čísla", fontsize=12)
for s in ("top", "right"):
    bx.spines[s].set_visible(False)

fig.text(0.6, 0.15, f"NumPy:   pixels[{PY}, {PX}]   (nejdřív řádek y, pak sloupec x)", fontsize=11.5, family="monospace")
fig.text(0.6, 0.07, f"kernel:  src(ivec2({PX}, {PY}))   (nejdřív x, pak y)", fontsize=11.5, family="monospace")
fig.text(0.6, 0.90, "tvar pole: (výška, šířka, 4)", fontsize=12, weight="bold")
save(fig, "image_as_array.png")


# 3. Workgroups and threads ------------------------------------------------------
fig = plt.figure(figsize=(11, 4.4))
ax = fig.add_axes([0.02, 0.08, 0.5, 0.82])
CW, CH, G = 64, 40, 8
ax.set_xlim(-1, CW + 1)
ax.set_ylim(CH + 1, -4)
ax.axis("off")
ax.add_patch(Rectangle((0, 0), CW, CH, facecolor=LIGHT["blue"], edgecolor=BLUE, linewidth=2))
for gx in range(0, CW + 1, G):
    ax.plot([gx, gx], [0, CH], color=BLUE, linewidth=1)
for gy in range(0, CH + 1, G):
    ax.plot([0, CW], [gy, gy], color=BLUE, linewidth=1)
TGX, TGY = 3, 1
ax.add_patch(Rectangle((TGX * G, TGY * G), G, G, facecolor=ORANGE, alpha=0.6, edgecolor=ORANGE, linewidth=2.5))
ax.text(CW / 2, -1.5, "plátno 64 × 40 px, skupiny (workgroups) 8 × 8", ha="center", fontsize=12, weight="bold")
ax.text(TGX * G + G / 2, TGY * G + G / 2, "(3, 1)", ha="center", va="center", fontsize=10, weight="bold")

zx = fig.add_axes([0.58, 0.12, 0.38, 0.72])
zx.set_xlim(-0.6, G - 0.4)
zx.set_ylim(G - 0.4, -0.6)
zx.set_aspect("equal")
zx.axis("off")
zx.add_patch(Rectangle((-0.5, -0.5), G, G, facecolor=LIGHT["orange"], edgecolor=ORANGE, linewidth=2.5))
LX, LY = 5, 2
for j in range(G):
    for i in range(G):
        hit = (i, j) == (LX, LY)
        zx.plot(i, j, "o", markersize=11 if hit else 7, color="#C0392B" if hit else ORANGE)
zx.set_title("jedna skupina = 64 vláken, každé počítá 1 pixel", fontsize=11.5)
fig.text(0.58, 0.05, f"group_id = ({TGX}, {TGY}),  local_id = ({LX}, {LY})", fontsize=11, family="monospace")
fig.text(0.58, 0.0, f"pixel = group_id · 8 + local_id = ({TGX * G + LX}, {TGY * G + LY})", fontsize=11, family="monospace")
save(fig, "thread_grid.png")


# 4. Fragment shader against compute kernel ----------------------------------------
fig, ax = plt.subplots(figsize=(12, 3.9))
ax.set_xlim(0, 12)
ax.set_ylim(0, 3.9)
ax.axis("off")
ax.text(0.05, 3.7, "OpenGL, co už znáte:", fontsize=12, weight="bold", color=GREY, va="top")
steps = ["vrcholy", "vertex\nshader", "rasterizace", "fragment\nshader", "framebuffer"]
for i, s in enumerate(steps):
    x = 0.1 + i * 2.2
    fill = LIGHT["green"] if "fragment" in s else LIGHT["grey"]
    edge = GREEN if "fragment" in s else GREY
    ax.add_patch(FancyBboxPatch((x, 2.25), 1.7, 0.95, boxstyle="round,pad=0.02,rounding_size=0.08", facecolor=fill, edgecolor=edge, linewidth=2))
    ax.text(x + 0.85, 2.72, s, ha="center", va="center", fontsize=10.5)
    if i < len(steps) - 1:
        arrow(ax, x + 1.75, 2.72, x + 2.15, 2.72)
ax.text(0.05, 1.75, "Compute, co je nové:", fontsize=12, weight="bold", color=GREY, va="top")
steps2 = [("mřížka vláken\n(1 vlákno = 1 pixel)", LIGHT["grey"], GREY), ("kernel", LIGHT["blue"], BLUE), ("obrázek\n(vrstva)", LIGHT["grey"], GREY)]
for i, (s, fill, edge) in enumerate(steps2):
    x = 0.1 + i * 3.3
    ax.add_patch(FancyBboxPatch((x, 0.3), 2.6, 0.95, boxstyle="round,pad=0.02,rounding_size=0.08", facecolor=fill, edgecolor=edge, linewidth=2))
    ax.text(x + 1.3, 0.77, s, ha="center", va="center", fontsize=10.5)
    if i < len(steps2) - 1:
        arrow(ax, x + 2.65, 0.77, x + 3.25, 0.77)
ax.text(9.55, 0.77, "žádné trojúhelníky,\nčte a zapisuje\nlibovolné pixely", ha="left", va="center", fontsize=10.5, color=BLUE, style="italic")
save(fig, "fragment_vs_compute.png")


# 5. The journey of a %%gmacs cell -----------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 4.0))
ax.set_xlim(0, 11)
ax.set_ylim(0, 4.0)
ax.axis("off")
row1 = [("buňka %%gmacs\nvaše funkce pixel", LIGHT["blue"], BLUE), ("Sára ji obalí\nna celý kernel", LIGHT["grey"], GREY),
        ("gmacs → GLSL 450", LIGHT["grey"], GREY), ("glslang →\nSPIR-V", LIGHT["grey"], GREY)]
row2 = [("zmenšený náhled\nv notebooku", LIGHT["blue"], BLUE), ("nová vrstva\nv Sáře", LIGHT["purple"], PURPLE),
        ("GPU spustí mřížku\na změří čas běhu", LIGHT["orange"], ORANGE)]
for i, (s, fill, edge) in enumerate(row1):
    x = 0.1 + i * 2.75
    ax.add_patch(FancyBboxPatch((x, 2.55), 2.2, 1.05, boxstyle="round,pad=0.02,rounding_size=0.08", facecolor=fill, edgecolor=edge, linewidth=2))
    ax.text(x + 1.1, 3.07, s, ha="center", va="center", fontsize=10.5)
    if i < len(row1) - 1:
        arrow(ax, x + 2.25, 3.07, x + 2.7, 3.07)
xs2 = [2.85, 5.6, 8.35]
for (s, fill, edge), x in zip(row2, xs2):
    ax.add_patch(FancyBboxPatch((x, 0.45), 2.2, 1.05, boxstyle="round,pad=0.02,rounding_size=0.08", facecolor=fill, edgecolor=edge, linewidth=2))
    ax.text(x + 1.1, 0.97, s, ha="center", va="center", fontsize=10.5)
arrow(ax, 9.45, 2.5, 9.45, 1.55)
arrow(ax, 8.3, 0.97, 7.85, 0.97)
arrow(ax, 5.55, 0.97, 5.1, 0.97)
ax.text(0.2, 1.0, "posuvník pošle jen\nnové hodnoty, kernel\nse znovu nepřekládá", fontsize=10.5, color=GREY, style="italic", va="center")
save(fig, "cell_journey.png")


# 6. The invert slider ----------------------------------------------------------------
c = np.linspace(0, 1, 100)
fig, ax = plt.subplots(figsize=(6.2, 4.2))
for t, col in ((0.0, BLUE), (0.5, GREY), (1.0, ORANGE)):
    ax.plot(c, c + t * (1 - 2 * c), linewidth=2.5, color=col, label=f"amount = {t:g}")
ax.set_xlabel("původní hodnota kanálu  c", fontsize=11)
ax.set_ylabel("nová hodnota  c'", fontsize=11)
ax.set_title("c' = mix(c, 1 − c, amount)", fontsize=12)
ax.legend(frameon=False, fontsize=10.5)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")
ax.grid(alpha=0.3)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
save(fig, "mix_graph.png")



# 7. Pixels between Sara and NumPy ----------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 4.6))
ax.set_xlim(0, 11)
ax.set_ylim(0, 4.6)
ax.axis("off")
box(ax, 0.1, 0.2, 2.7, 3.3, "Python", ["notebook", "pole NumPy", "float32, (h, w, 4)"], ORANGE, LIGHT["orange"])
box(ax, 8.2, 0.2, 2.7, 3.3, "Sára", ["vrstva v paměti GPU", "RGBA32F nebo RGBA16F"], BLUE, LIGHT["blue"])
ax.add_patch(FancyBboxPatch((4.15, 0.55), 2.7, 1.55, boxstyle="round,pad=0.02,rounding_size=0.08",
                            linewidth=2, edgecolor=GREY, facecolor=LIGHT["grey"]))
ax.text(5.5, 1.82, "dočasný soubor .npy", ha="center", va="center", fontsize=12, weight="bold", color="#222")
ax.text(5.5, 1.38, "hlavička + surová data", ha="center", va="center", fontsize=10.5, color="#333")
ax.text(5.5, 1.0, "1920 × 1080 px ≈ 33 MB", ha="center", va="center", fontsize=10.5, color="#333")
# the door carries the requests and answers
arrow(ax, 2.85, 3.05, 8.15, 3.05, color=PURPLE)
arrow(ax, 8.15, 2.7, 2.85, 2.7, color=PURPLE)
ax.text(5.5, 3.2, "příkaz v JSON: read_pixels, write_pixels, …", ha="center", va="bottom", fontsize=10.5, color=PURPLE)
ax.text(5.5, 2.55, "odpověď: cesta k souboru", ha="center", va="top", fontsize=10.5, color=PURPLE)
ax.text(5.5, 4.35, "spojení agent door: Unix socket na Linuxu, TCP s tokenem na Windows", ha="center", va="center", fontsize=11.5,
        weight="bold", color=PURPLE)
# pixels travel through the file
arrow(ax, 8.15, 1.65, 6.9, 1.65, color=BLUE)
arrow(ax, 4.1, 1.65, 2.85, 1.65, color=BLUE)
ax.text(7.52, 1.75, "GPU → RAM", ha="center", va="bottom", fontsize=9.5, color=BLUE)
ax.text(3.47, 1.75, "np.load", ha="center", va="bottom", fontsize=9.5, color=BLUE, family="monospace")
arrow(ax, 2.85, 0.85, 4.1, 0.85, color=ORANGE)
arrow(ax, 6.9, 0.85, 8.15, 0.85, color=ORANGE)
ax.text(3.47, 0.75, "np.save", ha="center", va="top", fontsize=9.5, color=ORANGE, family="monospace")
ax.text(7.52, 0.75, "RAM → GPU", ha="center", va="top", fontsize=9.5, color=ORANGE)
ax.text(0.15, 3.85, "layer.read()", fontsize=10.5, color=BLUE, family="monospace")
ax.text(0.15, 3.6, "doc.new_layer(...)", fontsize=10.5, color=ORANGE, family="monospace")
save(fig, "transport.png")


# 8. Mean against OKLab lightness ---------------------------------------------------------
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

swatches = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=float)
labels = ["červená", "zelená", "modrá", "žlutá", "azurová", "purpurová"]
mean = np.repeat(swatches.mean(axis=1, keepdims=True), 3, axis=1)
lab = np.cbrt(srgb_to_linear(swatches) @ M1.T) @ M2.T
oklab_grey = np.repeat(linear_to_srgb(lab[:, :1] ** 3), 3, axis=1)
rows = [("původní barva", swatches, None), ("průměr (R + G + B) / 3", mean, mean[:, 0]),
        ("OKLab, a = b = 0", oklab_grey, lab[:, 0])]

fig, ax = plt.subplots(figsize=(11, 3.6))
ax.set_xlim(-3.2, 6)
ax.set_ylim(3.1, -0.45)
ax.axis("off")
for r, (name, colours, numbers) in enumerate(rows):
    ax.text(-0.15, r + 0.4, name, ha="right", va="center", fontsize=11.5)
    for i, colour in enumerate(colours):
        ax.add_patch(Rectangle((i + 0.05, r + 0.05), 0.9, 0.7, facecolor=np.clip(colour, 0, 1), edgecolor="#BBBBBB"))
        if numbers is not None:
            ink = "black" if colour[0] > 0.45 else "white"
            ax.text(i + 0.5, r + 0.4, f"{numbers[i]:.2f}", ha="center", va="center", fontsize=10, color=ink)
for i, name in enumerate(labels):
    ax.text(i + 0.5, -0.12, name, ha="center", va="bottom", fontsize=10.5)
fig.text(0.5, 0.0, "čísla: hodnota šedé v řádku průměru, světlost L v řádku OKLab", ha="center", fontsize=10, color=GREY)
save(fig, "grey_comparison.png")


# 9. The test picture -------------------------------------------------------
# Not a diagram but the layer every effect runs on, so it is written pixel for
# pixel rather than drawn: 8-bit sRGB, 1920 by 1080, three bands of 360 rows.
W, H, BAND = 1920, 1080, 360
chart = np.zeros((H, W, 3))

# Top band: the six pure colours of the grey comparison, in its order.
for i, colour in enumerate(swatches):
    chart[0:BAND, i * W // 6:(i + 1) * W // 6] = colour

# Middle band: every hue at full saturation, red at both ends.
hue = np.arange(W) / W * 6.0
sector = np.floor(hue).astype(int) % 6
rise = hue - np.floor(hue)
fall = 1.0 - rise
one, zero = np.ones(W), np.zeros(W)
red = np.choose(sector, [one, fall, zero, zero, rise, one])
green = np.choose(sector, [rise, one, one, fall, zero, zero])
blue = np.choose(sector, [zero, zero, rise, one, one, fall])
chart[BAND:2 * BAND] = np.stack([red, green, blue], axis=-1)[None, :, :]

# Bottom band: grey from black on the left to white on the right.
chart[2 * BAND:] = (np.arange(W) / (W - 1))[None, :, None]

plt.imsave(os.path.join(OUT, "color_chart.png"), np.round(chart * 255).astype(np.uint8))

print(sorted(os.listdir(OUT)))
