"""Builds the lesson 3 notebook in the lecture folder above this one.

The notebook is generated: edit this file and run it again, never the .ipynb
by hand, or the next build overwrites the edit. The pictures and the test photo
coffee.png come from lesson_03_diagrams.py, which enlarges scikit-image's
coffee photo (CC0) twice and draws three objects in place of its wood on the
right, as imgs/3/SOURCES.md says.

Markdown cells are raw strings, so LaTeX keeps single backslashes and braces.
A picture from imgs/3 is written as @img(file, width, alt text).

The solutions are constants near the top, so the markdown shows exactly the
code that test_lesson_03.py runs.

The lesson opens with a brainstorm of how to blur and how to find edges, then
goes from the mean of the whole photo to a square around each pixel. Separability
is explained by the associativity of convolution. Sobel is a finished kernel to play
with. The lesson is short, because lesson 4 follows in the same slot. The NumPy port
in section 5 is given and is homework, its only task is copying it into Krita. The
bilateral filter is an optional bonus after it, and so are the edge rules, the shared
memory and the second derivative.
"""
import re
import textwrap
from pathlib import Path

import numpy as np

import lesson_writer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "lesson_03_convolution.ipynb"
IMG = "imgs/3"
WIDTH, HEIGHT = 1200, 800     # the photo's size and the canvas's
SIGMA = 3.0                   # every blur's opening sigma
cells = []


def _pictures(text):
    return re.sub(r"@img\(([^,]+), (\d+), ([^)]*)\)",
                  lambda m: f'<img src="{IMG}/{m[1]}" width="{m[2]}" alt="{m[3]}">', text)


def md(text):
    cells.extend(lesson_writer.markdown(_pictures(text)))


def code(text, hidden=False, solve=()):
    metadata = {"jupyter": {"source_hidden": True}} if hidden else {}
    cell = {"cell_type": "code", "metadata": metadata, "execution_count": None, "outputs": [],
            "source": lesson_writer.lines(text)}
    if solve:
        cell["solution"] = lesson_writer.solved(text, solve)
    cells.append(cell)


# --- The kernels ------------------------------------------------------------------
# The goal shows one effect of each of the brainstorm's three questions. It is not the code
# the lesson writes: one kernel instead of the NumPy filter, its blur in two passes through
# shared memory as in section 3, so its sliders move smoothly.
GOAL = """\
%%gmacs "coffee" -> "Goal" effect=0 blur_size=3 tolerance=0.1 noisy_photo=false
kernel goal

import linear_srgb_color_space
import oklab_color_space
import noise

image src: readonly
image dst

params:
    int effect: hint_enum("Blur", "Edges", "Denoise")
    float blur_size: hint_range(0.5, 6)
    float tolerance: hint_range(0.02, 0.3)
    bool noisy_photo

const int BLUR = 0
const int EDGES = 1
const int DENOISE = 2
const int WIDE = 8             # a workgroup is a column of 8 x 32 pixels
const int TALL = 32
const int R_MAX = 15           # the largest reach, 2.5 blur_size at blur_size 6

# The first pass, along x, for the group's columns and up to R_MAX rows above and below it.
shared vec3 across[(TALL + 2 * R_MAX) * WIDE]
shared vec3 across_lab[(TALL + 2 * R_MAX) * WIDE]


def photo(ivec2 p) -> vec3:
    \"\"\"The photo at p in linear light, with grain added when noisy_photo is on.\"\"\"
    p = clamp(p, ivec2(0), image_size(src) - 1)
    vec3 c = src[p].rgb
    if noisy_photo:
        vec3 grain = vec3(sa_hash22(vec2(p)), sa_hash21(vec2(p) + vec2(37.0, 11.0))) - 0.5
        c = clamp(c + 0.3 * grain, 0.0, 1.0)
    return srgb_to_linear_srgb(c)


def alike(vec3 a, vec3 b) -> float:
    \"\"\"How alike two OKLab colours are, 1 for the same and towards 0 past tolerance.\"\"\"
    vec3 d = a - b
    return exp(-dot(d, d) / (2.0 * tolerance * tolerance))


def blur(ivec2 at, ivec2 me, bool keep_edges) -> vec3:
    \"\"\"The Gaussian mean around at, the group's pixel me, out to 2.5 blur_size, along x into
    shared memory, then along y. With keep_edges a neighbour also weighs less the more its colour
    differs.\"\"\"
    int r = min(int(ceil(2.5 * blur_size)), R_MAX)
    ivec2 corner = at - me
    int first = (R_MAX - r) * WIDE
    # Pass 1: every row the group reaches, r more above and below, each thread a few of them.
    for i in range(me.y * WIDE + me.x, (TALL + 2 * r) * WIDE, WIDE * TALL):
        ivec2 p = corner + ivec2(i % WIDE, i / WIDE - r)
        vec3 centre = linear_srgb_to_oklab(photo(p))
        vec3 total = vec3(0.0)
        float weight = 0.0
        for dx in range_inclusive(-r, r):
            vec3 c = photo(p + ivec2(dx, 0))
            float k = exp(-float(dx * dx) / (2.0 * blur_size * blur_size))
            if keep_edges:
                k *= alike(linear_srgb_to_oklab(c), centre)
            total += k * c
            weight += k
        across[first + i] = total / weight
        across_lab[first + i] = linear_srgb_to_oklab(total / weight)
    barrier()
    # Pass 2: the first pass's rows above and below the pixel, along y.
    int mine = (me.y + R_MAX) * WIDE + me.x
    vec3 total = vec3(0.0)
    float weight
    for dy in range_inclusive(-r, r):
        int i = mine + dy * WIDE
        float k = exp(-float(dy * dy) / (2.0 * blur_size * blur_size))
        if keep_edges:
            k *= alike(across_lab[i], across_lab[mine])
        total += k * across[i]
        weight += k
    return linear_srgb_to_srgb(total / weight)


def edges(ivec2 at) -> vec3:
    \"\"\"The strength of the edge through at, from the lightness of its 3 x 3 neighbours, white on black.\"\"\"
    vec2 g = vec2(0.0)
    for j in range_inclusive(-1, 1):
        for i in range_inclusive(-1, 1):
            float light = linear_srgb_to_oklab(photo(at + ivec2(i, j))).x
            g += light * vec2(i * (2 - abs(j)), j * (2 - abs(i)))
    return vec3(clamp(2.0 * length(g), 0.0, 1.0))


@workgroup_size(8, 32)
def goal():
    match effect:
        case BLUR:
            dst[pixel] = vec4(blur(pixel, ivec2(local_id.xy), false), 1.0)
        case EDGES:
            dst[pixel] = vec4(edges(pixel), 1.0)
        case DENOISE:
            dst[pixel] = vec4(blur(pixel, ivec2(local_id.xy), true), 1.0)"""

# Section 2's convolution reads the kernel its drop-down picks: the student's own 3 x 3, mirrored
# from a 2 x 2 grid, or a larger preset. Each preset is its radius here and its weights in
# preset_weights, whose sums the kernel divides by.
PRESETS = {
    "Own 3 x 3": 1,
    "Box 15 x 15": 7,
    "Disc 25 x 25": 12,
    "Gauss 25 x 25": 12,
    "Motion 25 x 1": 12,
    "Sharpen 9 x 9": 4,
    "Edges 5 x 5": 2,
    "Emboss 5 x 5": 2,
}
OWN = [0.0, -1.0, -1.0, 5.0]     # the own grid's opening corner, sharpen: 0 -1 0 / -1 5 -1 / 0 -1 0
PRESET_SIGMA = 4.0               # the Gauss preset's sigma, 3 sigma is its radius


def preset_weights(name, own=OWN):
    """A preset's weights, (2r + 1, 2r + 1) with row j and column i from -r, divided by their sum
    unless it is 0, as the kernel's weight(i, j) answers them."""
    r = PRESETS[name]
    j, i = np.mgrid[-r:r + 1, -r:r + 1]
    centre = ((i == 0) & (j == 0)).astype(float)
    w = {
        "Own 3 x 3": lambda: np.array(own, float)[(1 - abs(j)) * 2 + (1 - abs(i))],
        "Box 15 x 15": lambda: np.ones(i.shape),
        "Disc 25 x 25": lambda: (i * i + j * j <= r * r).astype(float),
        "Gauss 25 x 25": lambda: np.exp(-(i * i + j * j) / (2 * PRESET_SIGMA ** 2)),
        "Motion 25 x 1": lambda: (j == 0).astype(float),
        "Sharpen 9 x 9": lambda: 3.0 * centre - 2.0 / 81,
        "Edges 5 x 5": lambda: np.where(centre > 0, 24.0, -1.0),
        "Emboss 5 x 5": lambda: (i + j) / 2 + centre,
    }[name]()
    total = w.sum()
    return w / total if abs(total) > 0.001 else w


CONV_STUB = """\
    # TODO: the neighbours out to reach(), each times its weight, summed
    return src[at]"""

SOLUTION_CONV = """\
    int r = reach()
    vec3 total = vec3(0.0)
    for j in range_inclusive(-r, r):
        for i in range_inclusive(-r, r):
            total += weight(i, j) * src[at - ivec2(i, j)].rgb
    return vec4(total, src[at].a)"""

_NAMES = ", ".join(f'"{name}"' for name in PRESETS)
_CONSTS = "\n".join(f"const int {name.split()[0].upper()} = {n}" for n, name in enumerate(PRESETS))
_REACH = "\n".join(f"        case {name.split()[0].upper()}:\n            return {r}" for name, r in PRESETS.items())
_DISC = int((preset_weights("Disc 25 x 25") > 0).sum())       # pixels inside the disc
_GAUSS_SUM = float(np.exp(-np.arange(-12, 13) ** 2 / (2 * PRESET_SIGMA ** 2)).sum() ** 2)

CONV = f"""\
%%gmacs "coffee" -> "Convolution"
uniform int preset: hint_enum({_NAMES}) = 0
uniform float own[4] = {{
     {OWN[0]:.1f}, {OWN[1]:.1f},
    {OWN[2]:.1f},  {OWN[3]:.1f}}}

{_CONSTS}


def reach() -> int:
    \"\"\"The kernel's radius r: it reads r pixels to each side.\"\"\"
    match preset:
{_REACH}
    return 0


def weight(int i, int j) -> float:
    \"\"\"The weight at the shift (i, j), divided by the sum of all the weights unless that is 0.\"\"\"
    bool centre = i == 0 and j == 0
    float own_sum = 4.0 * own[0] + 2.0 * own[1] + 2.0 * own[2] + own[3]
    match preset:
        case OWN:
            float w = own[(1 - abs(j)) * 2 + (1 - abs(i))]   # the grid is the top left corner, mirrored
            if abs(own_sum) > 0.001:
                return w / own_sum
            return w
        case BOX:
            return 1.0 / 225.0
        case DISC:
            if i * i + j * j <= 144:
                return 1.0 / {_DISC}.0
            return 0.0
        case GAUSS:
            return exp(-float(i * i + j * j) / {2 * PRESET_SIGMA ** 2:.1f}) / {_GAUSS_SUM:.4f}
        case MOTION:
            if j == 0:
                return 1.0 / 25.0
            return 0.0
        case SHARPEN:
            if centre:
                return 3.0 - 2.0 / 81.0
            return -2.0 / 81.0
        case EDGES:
            if centre:
                return 24.0
            return -1.0
        case EMBOSS:
            if centre:
                return 1.0
            return float(i + j) / 2.0
    return 0.0


def pixel(ivec2 at) -> vec4:
{{CONV_STUB}}"""
CONV = CONV.replace("{CONV_STUB}", CONV_STUB)

# Section 2's question of the colour space, on the photo's saturated yellow sun in a blue sky.
SPACES = """\
%%gmacs "coffee" -> "Spaces" halo=8
import linear_srgb_color_space
import oklab_color_space
uniform int effect: hint_enum("Blur", "Sharpen") = 1
uniform int space: hint_enum("sRGB", "Linear RGB", "OKLab") = 0
uniform int radius: hint_range(1, 8) = 4
uniform float sharpen: hint_range(0, 4) = 2.0

const int BLUR = 0
const int SHARPEN = 1
const int SRGB = 0
const int LINEAR = 1
const int OKLAB = 2


def into(vec3 c) -> vec3:
    \"\"\"The sRGB colour c in the space the drop-down picks.\"\"\"
    match space:
        case SRGB:
            return c
        case LINEAR:
            return srgb_to_linear_srgb(c)
        case OKLAB:
            return linear_srgb_to_oklab(srgb_to_linear_srgb(c))


def back(vec3 c) -> vec3:
    \"\"\"A colour of that space in sRGB again, cut to what the screen shows.\"\"\"
    match space:
        case SRGB:
            return clamp(c, 0.0, 1.0)
        case LINEAR:
            return linear_srgb_to_srgb(clamp(c, 0.0, 1.0))
        case OKLAB:
            return linear_srgb_to_srgb(clamp(oklab_to_linear_srgb(c), 0.0, 1.0))


def pixel(ivec2 at) -> vec4:
    int side = 2 * radius + 1
    vec3 mean = vec3(0.0)
    for dy in range(side):
        for dx in range(side):
            mean += into(src[at + ivec2(dx - radius, dy - radius)].rgb)
    mean /= float(side * side)
    vec3 c = into(src[at].rgb)
    vec3 sharp = c + sharpen * (c - mean)
    if space == OKLAB:
        sharp = vec3(sharp.x, c.yz)              # in OKLab only the lightness L, the colour stays
    match effect:
        case BLUR:
            return vec4(back(mean), 1.0)
        case SHARPEN:
            return vec4(back(sharp), 1.0)"""

GAUSS_2D = """\
%%gmacs "coffee" -> "Gauss 2D" halo=32
import linear_srgb_color_space

uniform float sigma: hint_range(0.5, 8) = 3.0


# Every neighbour within 3 sigma, weighted by the Gaussian, in linear light.
def pixel(ivec2 at) -> vec4:
    int r = int(ceil(3.0 * sigma))
    vec3 total = vec3(0.0)
    float weight = 0.0
    for dy in range_inclusive(-r, r):
        for dx in range_inclusive(-r, r):
            float k = exp(-float(dx * dx + dy * dy) / (2.0 * sigma * sigma))
            total += k * srgb_to_linear_srgb(src[at + ivec2(dx, dy)].rgb)
            weight += k
    return vec4(linear_srgb_to_srgb(total / weight), 1.0)"""

BLUR_X_STUB = """\
    # TODO: the Gaussian sum along the row, divided by the sum of the weights
    return vec4(srgb_to_linear_srgb(src[at].rgb), 1.0)"""

SOLUTION_BLUR_X = """\
    int r = int(ceil(3.0 * sigma))
    vec3 total = vec3(0.0)
    float weight = 0.0
    for dx in range_inclusive(-r, r):
        float k = exp(-float(dx * dx) / (2.0 * sigma * sigma))
        total += k * srgb_to_linear_srgb(src[at + ivec2(dx, 0)].rgb)
        weight += k
    return vec4(total / weight, 1.0)"""

BLUR_X = f"""\
%%gmacs "coffee" -> "Blur X" halo=32
import linear_srgb_color_space

uniform float sigma: hint_range(0.5, 8) = 3.0


# The first pass, along the row. It keeps linear light in the layer, which the
# second pass reads as it is.
def pixel(ivec2 at) -> vec4:
{BLUR_X_STUB}"""

BLUR_Y = """\
%%gmacs "Blur X" -> "Blur Y" halo=32
import linear_srgb_color_space

uniform float sigma: hint_range(0.5, 8) = 3.0


# The second pass, along the column of what the first pass wrote, in linear
# light already, so only the result is encoded to sRGB.
def pixel(ivec2 at) -> vec4:
    int r = int(ceil(3.0 * sigma))
    vec3 total = vec3(0.0)
    float weight = 0.0
    for dy in range_inclusive(-r, r):
        float k = exp(-float(dy * dy) / (2.0 * sigma * sigma))
        total += k * src[at + ivec2(0, dy)].rgb
        weight += k
    return vec4(linear_srgb_to_srgb(total / weight), 1.0)"""

BILATERAL_STUB = """\
            # TODO: times the colour weight, a Gaussian of how far lab(p) lies from centre"""

SOLUTION_BILATERAL = """\
            vec3 d = lab(p) - centre
            k *= exp(-dot(d, d) / (2.0 * sigma_c * sigma_c))"""

BILATERAL = f"""\
%%gmacs "coffee" -> "Bilateral" halo=16
import linear_srgb_color_space
import oklab_color_space

uniform float sigma: hint_range(0.5, 4) = 3.0
uniform float sigma_c: hint_range(0.01, 0.5) = 0.05


def lab(ivec2 p) -> vec3:
    return linear_srgb_to_oklab(srgb_to_linear_srgb(src[p].rgb))


# Gauss 2D, with each neighbour's weight also falling with how far its colour lies from the centre's.
def pixel(ivec2 at) -> vec4:
    int r = int(ceil(3.0 * sigma))
    vec3 centre = lab(at)
    vec3 total = vec3(0.0)
    float weight = 0.0
    for dy in range_inclusive(-r, r):
        for dx in range_inclusive(-r, r):
            ivec2 p = at + ivec2(dx, dy)
            float k = exp(-float(dx * dx + dy * dy) / (2.0 * sigma * sigma))
{BILATERAL_STUB}
            total += k * srgb_to_linear_srgb(src[p].rgb)
            weight += k
    return vec4(linear_srgb_to_srgb(total / weight), 1.0)"""

SOLUTION_FETCH = """\
        case CLAMP:
            return src[clamp(p, ivec2(0), SIZE - 1)]
        case MIRROR:
            return src[min(max(p, -1 - p), 2 * SIZE - 1 - p)]
        case WRAP:
            return src[p - SIZE * ivec2(floor(vec2(p) / vec2(SIZE)))]"""

SHIFT = (160, 96)             # the edge kernel's opening shift, dx and dy

EDGES = f"""\
%%gmacs "coffee" -> "Shifted" halo=256
uniform int mode: hint_enum("Zero", "Clamp", "Mirror", "Wrap") = 3
uniform int dx: hint_range(-256, 256) = {SHIFT[0]}
uniform int dy: hint_range(-256, 256) = {SHIFT[1]}

const int ZERO = 0
const int CLAMP = 1
const int MIRROR = 2
const int WRAP = 3
const ivec2 SIZE = ivec2({WIDTH}, {HEIGHT})


def fetch(ivec2 p) -> vec4:
    \"\"\"The pixel at p, wherever p lies, by the edge rule mode picks.\"\"\"
    bool inside = p.x >= 0 and p.y >= 0 and p.x < SIZE.x and p.y < SIZE.y
    match mode:
        case ZERO:
            if inside:
                return src[p]
            return vec4(0.0, 0.0, 0.0, 1.0)
{SOLUTION_FETCH}
    return vec4(0.0, 0.0, 0.0, 1.0)


def pixel(ivec2 at) -> vec4:
    return fetch(at - ivec2(dx, dy))"""

SHARED = """\
%%gmacs "coffee" -> "Blur X shared" halo=32 sigma=3
kernel blur_x_shared

import linear_srgb_color_space

image src: readonly
image dst

params:
    float sigma: hint_range(0.5, 8)

const int GROUP = 64          # threads in a workgroup, one row of 64 pixels
const int R_MAX = 24          # the largest radius, 3 sigma at sigma 8

# One row of the group's pixels and R_MAX more on each side, in linear light.
shared vec3 row[GROUP + 2 * R_MAX]


@workgroup_size(64, 1)
def blur_x_shared():
    ivec2 size = image_size(src)
    # Step 1: the group's threads load the row together, one or two pixels each.
    int first = int(group_id.x) * GROUP - R_MAX
    for i in range(int(local_id.x), GROUP + 2 * R_MAX, GROUP):
        ivec2 p = clamp(ivec2(first + i, pixel.y), ivec2(0), size - 1)
        row[i] = srgb_to_linear_srgb(src[p].rgb)
    # Step 2: wait until every thread of the group has written its part.
    barrier()
    # Step 3: the Gaussian sum, read from shared memory.
    int r = min(int(ceil(3.0 * sigma)), R_MAX)
    vec3 total = vec3(0.0)
    float weight = 0.0
    for dx in range_inclusive(-r, r):
        float k = exp(-float(dx * dx) / (2.0 * sigma * sigma))
        total += k * row[int(local_id.x) + R_MAX + dx]
        weight += k
    dst[pixel] = vec4(total / weight, 1.0)"""

SOBEL_BODY = """\
    vec2 g = vec2(0.0)
    for j in range_inclusive(-1, 1):
        for i in range_inclusive(-1, 1):
            g += light(at + ivec2(i, j)) * vec2(i * (2 - abs(j)), j * (2 - abs(i)))"""

SOBEL = f"""\
%%gmacs "coffee" -> "Sobel"
import linear_srgb_color_space
import oklab_color_space

uniform float gain: hint_range(0.25, 8) = 2.0
uniform bool direction = false


def light(ivec2 p) -> float:
    return linear_srgb_to_oklab(srgb_to_linear_srgb(src[p].rgb)).x


def shade(vec2 g) -> vec3:
    \"\"\"The edge's strength as lightness and, with direction on, its direction as hue.\"\"\"
    float m = clamp(gain * length(g), 0.0, 1.0)
    if not direction or m == 0.0:
        return vec3(m)
    vec2 ab = 0.15 * m * normalize(g)
    return linear_srgb_to_srgb(max(oklab_to_linear_srgb(vec3(0.85 * m, ab)), vec3(0.0)))


def pixel(ivec2 at) -> vec4:
{SOBEL_BODY}
    return vec4(shade(g), 1.0)"""

# --- NumPy ------------------------------------------------------------------------
HELPERS_NP = '''\
def srgb_to_linear(c):
    a = np.abs(c)
    return np.sign(c) * np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    a = np.abs(c)
    return np.sign(c) * np.where(a <= 0.0031308, a * 12.92, 1.055 * a ** (1 / 2.4) - 0.055)


M1 = np.array([                      # linear RGB to LMS
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
], dtype=np.float32)
M2 = np.array([                      # cube-rooted LMS to OKLab
    [0.2104542553,  0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050,  0.4505937099],
    [0.0259040371,  0.7827717662, -0.8086757660],
], dtype=np.float32)
M1_INV = np.linalg.inv(M1)
M2_INV = np.linalg.inv(M2)


def linear_to_oklab(rgb):
    return np.cbrt(rgb @ M1.T) @ M2.T


def oklab_to_linear(lab):
    return ((lab @ M2_INV.T) ** 3) @ M1_INV.T


def gaussian_1d(sigma):
    """The weights of a Gaussian of sigma pixels out to 3 sigma, summing to one."""
    r = int(np.ceil(3 * sigma))
    x = np.arange(-r, r + 1)
    k = np.exp(-x ** 2 / (2 * sigma ** 2))
    return k / k.sum()'''

SOLUTION_BLUR_NP = '''\
def blur(a, sigma):
    """a, (h, w, c), blurred by a Gaussian of sigma pixels: rows, then columns, edges clamped."""
    k = gaussian_1d(sigma)
    r = len(k) // 2
    h, w = a.shape[:2]
    p = np.pad(a, ((0, 0), (r, r), (0, 0)), mode="edge")
    a = sum(k[i] * p[:, i:i + w] for i in range(len(k)))
    p = np.pad(a, ((r, r), (0, 0), (0, 0)), mode="edge")
    return sum(k[i] * p[i:i + h] for i in range(len(k)))'''

PARAMS_NP = '''\
PARAMS = {
    "mode": ["Blur", "Sharpen"],
    "sigma": (3.0, 0.5, 16.0),
    "sharpen": (1.0, 0.0, 4.0),
}'''

SOLUTION_APPLY = '''\
def apply(img, mode, sigma, sharpen):
    out = img.copy()
    lin = srgb_to_linear(img[..., :3])
    if mode == "Blur":
        lin = blur(lin, sigma)
    else:
        lab = linear_to_oklab(lin)
        light = lab[..., :1]                         # OKLab L, the colour stays
        lab[..., :1] = light + sharpen * (light - blur(light, sigma))
        lin = np.maximum(oklab_to_linear(lab), 0.0)
    out[..., :3] = linear_to_srgb(lin)
    return out'''

def answer(snippet, lang="python"):
    """A solution block's code fence."""
    return f"```{lang}\n{snippet}\n```"




# --- The interactive pictures of section 3 -------------------------------------------
# Hidden cells: what they draw is the lesson, how they draw it is not. No ipywidgets.Output,
# which VS Code cannot draw: the figure goes to an Image under its slider.
GAUSS_1D_PLOT = '''\
# Interactive picture: the Gaussian weights along a row, hidden because only the picture matters.
import io

import ipywidgets as widgets
from IPython.display import display
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle


def explore(draw, value=3.0):
    """draw(sigma)'s figure under a sigma slider, drawn again whenever the slider moves."""
    slider = widgets.FloatSlider(value=value, min=0.5, max=8.0, step=0.5, description="sigma",
                                 continuous_update=False)
    picture = widgets.Image(format="png", layout=widgets.Layout(max_width="100%", max_height="400px",
                                                                object_fit="contain"))

    def redraw(_=None):
        buffer = io.BytesIO()
        draw(slider.value).savefig(buffer, format="png", dpi=72, bbox_inches="tight", facecolor="white")
        picture.value = buffer.getvalue()

    slider.observe(redraw, names="value")
    redraw()
    display(widgets.VBox([slider, picture]))


def gauss_1d(sigma):
    r = int(np.ceil(3 * sigma))
    fig = Figure(figsize=(10, 3.4))
    ax = fig.subplots()
    x = np.linspace(-26, 26, 1041)
    ax.plot(x, np.exp(-x ** 2 / (2 * sigma ** 2)), color="#4C78A8", lw=2)
    i = np.arange(-r, r + 1)
    ax.vlines(i, 0, np.exp(-i ** 2 / (2 * sigma ** 2)), color="#F58518", lw=2.5)
    ax.axvspan(-r - 0.5, r + 0.5, color="#F58518", alpha=0.08)
    for at in (-3 * sigma, 3 * sigma):
        ax.axvline(at, color="#999999", lw=1, ls="--")
    ax.text(3 * sigma + 0.4, 0.9, "$3\\\\sigma$", color="#666666", fontsize=11)
    ax.set_title(f"$\\\\sigma$ = {sigma:g}: kernel sečte {2 * r + 1} sousedů v řádku, od {-r} do {r}", fontsize=12)
    ax.set_xlim(-26, 26)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("posun $i$ (pixely)")
    ax.set_ylabel("váha $k(i)$")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return fig


explore(gauss_1d)'''

GAUSS_2D_PLOT = '''\
# Interactive picture: the Gaussian weights of the square around a pixel, in the plane and as a hill.
def gauss_2d(sigma):
    r = int(np.ceil(3 * sigma))
    fig = Figure(figsize=(11, 4.4))
    flat = fig.add_subplot(1, 2, 1)
    j, i = np.mgrid[-26:27, -26:27]
    flat.imshow(np.exp(-(i * i + j * j) / (2 * sigma ** 2)), cmap="magma", vmin=0, vmax=1,
                extent=(-26.5, 26.5, 26.5, -26.5), interpolation="nearest")
    flat.add_patch(Rectangle((-r - 0.5, -r - 0.5), 2 * r + 1, 2 * r + 1, fill=False, ec="#F58518", lw=1.5))
    for k in (1, 2, 3):
        flat.add_patch(Circle((0, 0), k * sigma, fill=False, ec="white", lw=0.8, ls="--"))
    flat.set_title(f"váhy pixelů, čtverec {2 * r + 1} × {2 * r + 1}", fontsize=12)
    flat.set_xlabel("$i$")
    flat.set_ylabel("$j$")
    hill = fig.add_subplot(1, 2, 2, projection="3d")
    s = np.linspace(-26, 26, 105)
    x, y = np.meshgrid(s, s)
    hill.plot_surface(x, y, np.exp(-(x * x + y * y) / (2 * sigma ** 2)), cmap="magma", vmin=0, vmax=1,
                      rstride=1, cstride=1, linewidth=0)
    hill.set_zlim(0, 1)
    hill.set_title("tytéž váhy jako kopec", fontsize=12)
    return fig


explore(gauss_2d)'''


# --- Title --------------------------------------------------------------------
md(r"""
# Lekce 3: Konvoluce, rozmazání a hrany

Lekce odpovídá na tři otázky: jak fotku rozmazat, jak v ní najít hrany a jak z ní odstranit šum, aby hrany zůstaly ostré. Cestou napíšete konvoluci a Gaussovo rozmazání ve dvou průchodech a vyzkoušíte Sobelův detektor hran. Doma přepíšete rozmazání a zaostření do filtru pro Kritu. Lekce je krátká, hned za ní ve stejném bloku následuje lekce 4 o frekvencích. Rozmazání, které nechá hrany, a pravidla okraje jsou nepovinné.

Spusťte buňku. Otevře nový dokument a vloží do něj fotku jako vrstvu `coffee`. Vpravo má místo dřeva tři nakreslené předměty, na kterých se filtry liší nejvíc: žluté slunce na modré obloze, černou čáru pod úhlem 45° na bílé a drobná světla v noci.

<details><summary>📖 K čemu je plátno 1200 × 800</summary>

Je stejně velké jako fotka. Kernely z ní čtou a píšou do vrstev o stejné velikosti.
</details>
""")

code(f"""
import sys
sys.path.insert(0, "lib")            # sara.py and its helpers live in lib/
import numpy as np
import sara

doc, layer, pixels = sara.init(width={WIDTH}, height={HEIGHT})  # a canvas the size of the photo
photo = doc.import_image("imgs/3/coffee.png")            # a layer called "coffee"
""")

md(r"""
<details><summary>📖 Nové funkce v této lekci</summary>

**Kernel v buňce `%%gmacs`**

| Zápis | Co dělá | Sekce |
|---|---|---|
| `uniform float own[4]` | matice 2 × 2 jako mřížka políček pod buňkou, čtená po řádcích | 2 |
| `halo=32` na prvním řádku buňky | kernel smí číst až 32 pixelů za okrajem vrstvy | 3 |
| `min`, `max`, `clamp` na `ivec2`, `floor` na `vec2` a převod `ivec2(...)` | po složkách | 7, nepovinné |
| `kernel`, `image`, `params:`, `dst[pixel] = v` | celý kernel místo funkce `pixel`, s obrázky a parametry | nepovinné |
| `shared`, `barrier()`, `local_id`, `group_id` | sdílená paměť skupiny vláken, čekání na celou skupinu a čísla vlákna a skupiny | nepovinné |

**NumPy**

| Volání | Co dělá | Sekce |
|---|---|---|
| `np.pad(a, šířky, mode=...)` | pole rozšířené o okraje podle pravidla, `"edge"`, `"symmetric"`, `"wrap"` nebo `"constant"` | 5 |
| `a[:, i:i + w]` | výřez pole, pohled na stejná data bez kopie | 5 |

</details>
""")

md(r"""
## Cíl lekce

Cíl je jeden hotový kernel. Počítá jinak a rychleji než kód, který v lekci napíšete, aby se posuvníky hýbaly plynule. `effect` vybírá jeden ze tří efektů lekce. `blur_size` říká, jak daleko rozmazání sahá, a platí pro `Blur` i `Denoise`. `tolerance` platí jen pro `Denoise`: jak moc se smí barva souseda lišit, aby se ještě přimíchal. `noisy_photo` přidá do fotky zrno, na kterém je vidět, co `Denoise` umí.
""")

code(GOAL, hidden=True)

# --- Brainstorming --------------------------------------------------------------
md(r"""
## Jak byste to udělali?

Cíl umí tři věci a ke každé patří jedna otázka. U každé zkuste nejdřív vymyslet vlastní postup, než otevřete nápady.

### Rozmazání

V cíli zvolte `Blur` a měňte `blur_size`. Jak byste fotku rozmazali, když kernel vidí fotku a polohu svého pixelu?

<details><summary>💭 Nápady</summary>

Dobrý nápad splní tři věci: každý pixel dostane barvu namíchanou z okolí, výsledek se mění plynule a míru rozmazání jde nastavit. Každý pixel se navíc počítá sám z původní fotky, nezávisle na ostatních.

<details><summary>Průměr</summary>

Nehodí se. Průměr celé fotky dá každému pixelu stejnou barvu a z fotky zbude jedna skvrna. Průměr musí být jen z okolí pixelu.

@img(idea_mean_all.png, 520, Původní výřez fotky a průměr celé fotky: jedna hnědá barva)

</details>

<details><summary>Průměr po blocích</summary>

Rozmazání to není, vznikne mozaika (*pixelation*). Dva sousední pixely ze dvou bloků dostanou úplně jiný průměr a hranice bloků jsou ostré.

@img(idea_blocks.png, 520, Původní výřez fotky a mozaika z průměrů bloků)

</details>

<details><summary>Smíchat posunuté kopie fotky</summary>

Ano. Fotka smíchaná se svými posunutými kopiemi je rozmazaná. Pár kopií s velkým posunem dá dvojitý obraz, kopie posunuté o každý pixel dají hladké rozmazání. S různými vahami kopií je to konvoluce, sekce 2, a NumPy ji tak počítá, sekce 5.

@img(idea_shifted.png, 780, Původní výřez, průměr devíti kopií posunutých o 12 pixelů s mnohonásobným obrazem a průměr kopií posunutých o každý pixel do 8 s hladkým rozmazáním)

</details>

<details><summary>Průměr čtverce kolem každého pixelu</summary>

Ano, tomu se říká průměr okolí (*box blur*), sekce 1. Každý pixel má vlastní čtverec se sebou uprostřed. Čtverce dvou sousedních pixelů se skoro celé překrývají, a výsledek se proto mění plynule. Velikost čtverce říká, jak moc rozmazává.

@img(idea_box.png, 560, Zvětšené pixely fotky se čtverci 9 krát 9 kolem dvou sousedních pixelů A a B, které se skoro celé překrývají, a průměr čtverce 17 krát 17 kolem každého pixelu: rozmazaný obraz)

</details>

<details><summary>Barva náhodného souseda</summary>

Zajímavý efekt, ale rozmazání to není. Je to jako dithering z minulé lekce: rozmazaně vypadá jen z dálky, kde oko samo zprůměruje sousední pixely. Zblízka je obraz zrnitý jako za matným sklem.

@img(idea_random.png, 520, Původní výřez fotky a fotka, kde každý pixel vzal barvu náhodného souseda do 8 pixelů: zrnitý obraz)

</details>

<details><summary>Zmenšit fotku a zase zvětšit</summary>

Funguje a je ze všech nejrychlejší, protože obojí umí GPU v hardwaru. Textura si drží předem zmenšené kopie sebe sama (*mipmapy*) a při čtení mezi pixely sama smíchá čtyři nejbližší (bilineární filtrování). Rozmazání přes osm pixelů tak stojí jedno čtení na pixel místo stovek. Míru rozmazání ale jde měnit jen po skocích a při velkém zmenšení je vidět mřížka, ze které se zvětšovalo. Hry tak počítají rozmazání pro bloom, viz Co jsme vynechali.

@img(idea_resize.png, 780, Původní výřez fotky, jeho osminásobné zmenšení a zvětšení zpátky: rozmazaný obraz)

</details>

</details>

### Hrany

V cíli zvolte `Edges`. Jak byste ve fotce našli hrany?

<details><summary>💭 Nápady</summary>

Hrana je místo, kde se rychle mění světlost. Dobrý nápad najde hrany všech směrů, vyjde z něj síla hrany a neutopí se v šumu.

<details><summary>Oddělit světlá místa od tmavých</summary>

Nehodí se. Práh najde světlá místa, ne hrany. Hrana mezi dvěma tmavými plochami pod prahem zmizí, hrana mezi dvěma světlými nad ním taky.

@img(idea_threshold.png, 520, Světlost výřezu fotky a pixely nad prahem 0.6: světlé plochy, ne hrany)

</details>

<details><summary>Odečíst sousední pixely</summary>

Skoro. Rozdíl pravého a levého souseda je velký na svislých hranách a nula na vodorovných, potřebujeme tedy dva rozdíly, do strany a nahoru. Jeden řádek k tomu zachytí i šum, proto se rozdíl průměruje přes tři řádky. To je Sobelův operátor, sekce 4.

@img(idea_diff.png, 780, Světlost výřezu fotky, rozdíl vlevo a vpravo s hranami svislými a rozdíl nahoře a dole s hranami vodorovnými)

</details>

<details><summary>Porovnat pixel s průměrem jeho sousedů</summary>

Funguje. Na ploše je pixel stejný jako jeho sousedé a rozdíl vyjde nula, na hraně se liší. Najde hrany všech směrů najednou, ale ne jejich směr, a protože porovnává jednotlivé pixely, svítí i zrno a šum. Vyzkoušíte ho v sekci 2 jako jádro `Edges`.

@img(idea_neighbours.png, 640, Světlost výřezu a zesílený rozdíl každého pixelu od průměru jeho osmi sousedů, který svítí na hranách i v zrnité textuře)

</details>

<details><summary>Odečíst od fotky její rozmazanou kopii</summary>

Funguje. Rozmazání vyhladí detaily a rozdíl od něj zbude jen na hranách a jemné textuře. Je to předchozí nápad s větším okolím, a šum proto vadí méně. Dává sílu hrany, ale ne její směr.

@img(idea_dog.png, 780, Světlost výřezu fotky, její rozmazání a rozdíl obou, který svítí na hranách)

</details>

<details><summary>Ztenčit hrany na čáru jeden pixel širokou</summary>

Dobrý nápad na potom, až hrany najdeme. Z široké hrany se nechá jen pixel, kde je nejsilnější, a zbudou čisté čáry. Tak pracuje Cannyho detektor hran, v části Co jsme vynechali.

</details>

</details>

### Rozmazání, které nechá hrany

V cíli zapněte `noisy_photo` a porovnejte `Blur` a `Denoise`. `Blur` zrno vyhladí, ale rozmaže i okraj hrnku, `Denoise` nechá hrany ostré. Jak byste rozmazali šum, aby hrany zůstaly?

<details><summary>💭 Nápady</summary>

Dobrý nápad vyhladí plochy a šum a hrany nechá ostré.

<details><summary>Rozmazat jen tam, kde nejsou hrany</summary>

Jde to, hrany najde Sobel a jeho síla poslouží jako maska. Kolem hran ale zůstane zrno, protože tam se nerozmazává nic. Lepší je rozmazávat všude, jen nemíchat sousedy z druhé strany hrany.

@img(idea_mask.png, 780, Zašuměný výřez, síla hran jako maska a výsledek rozmazaný mimo hrany, kde kolem hran zůstalo zrno)

</details>

<details><summary>Vzít prostřední barvu okolí místo průměru</summary>

Funguje, je to medián. Prostřední hodnota z okolí hrany drží a zrno vyhladí, jenže plochy se změní v malované skvrny a na GPU se musí pro každý pixel třídit. Najdete ho v části Co jsme vynechali.

@img(idea_median.png, 520, Zašuměný výřez a medián okolí 7 krát 7: zrno zmizelo, hrany drží, plochy jsou skvrnité)

</details>

<details><summary>Míchat jen sousedy podobné barvy</summary>

Přesně tak. Soused za hranou má jinou barvu, a dostane proto malou váhu. To je bilaterální filtr, nepovinná sekce 6, a takhle počítá `Denoise` v cíli.

@img(idea_similar.png, 520, Zašuměný výřez a průměr jen sousedů podobné barvy: zrno zmizelo, okraj hrnku zůstal ostrý)

</details>

</details>
""")

# --- 1. Box blur --------------------------------------------------------------------
md(r"""
## 1. Rozmazání

Rozmazaná fotka vypadá jako z nezaostřeného objektivu: světlo každého bodu se rozlije do kruhové skvrny a skvrny sousedních bodů se sečtou. K takovému výsledku se chceme přiblížit. Světla vpravo dole i odlesky na lžičce se rozlijí do kulatých jasných kruhů s ostrým okrajem. Krita má tenhle filtr pod jménem Lens Blur.

@img(lens_blur.png, 600, Pravá část fotky se lžičkou, sluncem, šikmou čárou a světly, a tatáž část rozmazaná jako objektivem: světla a odlesky jsou kulaté jasné kruhy s ostrým okrajem)

<details><summary>📖 Jak vznikl obrázek objektivu</summary>

Každý pixel je průměr kruhu o poloměru 12 pixelů kolem sebe, spočítaný v lineárním světle. Odlesky jsou ve fotce oříznuté na bílou, ve skutečnosti byly mnohem jasnější. Obrázek je proto před rozmazáním zjasní, stejně jako světla v noci, jinak by se rozlily do slabých skvrn. Slunce a bílou za čárou nechá, jak jsou.
</details>

Nejjednodušší rozmazání je **průměr okolí** (*box blur*). Pixel dostane průměr čtverce o straně $2r + 1$ se sebou uprostřed:

$$g(x, y) = \frac{1}{(2r + 1)^2} \sum_{j=-r}^{r} \sum_{i=-r}^{r} f(x + i,\, y + j)$$

@img(box_window.png, 900, Okolí pixelu se šikmou hranou, okno 3 krát 3 se stejnými vahami jedna devítina a výsledek, ve kterém má přechod přes hranu pět mezistupňů)

Okno se posune na každý pixel a pixel dostane průměr svého okna. Na šikmé hraně je vidět, co čtverec dělá jinak než objektiv. Přes rovnou hranu by z okna 3 × 3 vyšly tři mezistupně, přes šikmou jich vyjde pět, protože čtverec sahá po úhlopříčce dál než do strany. Šikmá hrana se tak rozmaže víc než rovná a jasný bod se rozmaže do čtverce:

@img(box_vs_lens.png, 600, Pravá část fotky rozmazaná průměrem čtverce 25 krát 25 se čtvercovými světly a objektivem s kulatými)

Průměr okolí napíšete v sekci 2 jako jedno z jader konvoluce.
""")

# --- 2. Convolution ---------------------------------------------------------------
md(r"""
## 2. Konvoluce

Průměr dává všem sousedům stejnou váhu. S různými vahami, i zápornými, umí tentýž výpočet víc než rozmazat. **Konvoluce** (*convolution*) dvou funkcí $f$ a $g$ je

$$(f * g)(x, y) = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} g(u, v)\, f(x - u,\, y - v)\, \mathrm{d}u\, \mathrm{d}v$$

V každém bodě $(x, y)$ sečte hodnoty $f$ z okolí, každou vynásobenou vahou $g(u, v)$ podle toho, o kolik je posunutá. Obrázek má jen pixely, integrál je proto součet a váhy tvoří malou matici $w$ s poloměrem $r$, **jádro** (*kernel*, tady ve významu matice vah, ne programu na GPU):

$$(f * w)(x, y) = \sum_{j=-r}^{r} \sum_{i=-r}^{r} w(i, j)\, f(x - i,\, y - j)$$

Průměr okolí ze sekce 1 je konvoluce s jádrem, jehož váhy jsou všechny $1 / (2r + 1)^2$.

@img(convolution_window.png, 900, Okno 3 krát 3 nad mřížkou pixelů, každý pixel okna vynásobený svou váhou a součet zapsaný do pixelu uprostřed)

<details><summary>📖 Proč mínus ve vzorci</summary>

Mínus otočí jádro o 180°: váha $w(1, 0)$ patří sousedovi vlevo. Díky tomu je konvoluce komutativní, $f * g = g * f$, a asociativní, což využije sekce 3. Stejný součet s $f(x + i,\, y + j)$ se jmenuje korelace (*correlation*). Souměrná jádra, jako průměr nebo Gauss, vyjdou oběma způsoby stejně.
</details>

### 🎯 Úkol 1: konvoluce

Napište konvoluci podle vzorce výše. Poloměr jádra $r$ vrací `reach()`, váhu $w(i, j)$ funkce `weight(i, j)`, už vydělenou součtem všech vah. Soused $f(x - i,\, y - j)$ je `src[at - ivec2(i, j)]`. Nedokončený kernel vrací původní pixel.

Jádro vybírá `preset`. Hotová jádra jsou velká, aby byl efekt vidět i na velké fotce. `Own 3 x 3` je vaše jádro z mřížky `own`, jeho levý horní roh. Zbytek jádra se do něj zrcadlí podle os x a y:

$$\begin{pmatrix} a & b \\ c & d \end{pmatrix} \quad\longrightarrow\quad \begin{pmatrix} a & b & a \\ c & d & c \\ a & b & a \end{pmatrix}$$

Výchozí hodnoty mřížky dají zaostření.

<details><summary>💡 Nápověda</summary>

1. `int r = reach()`, pak dva vnořené cykly přes `j` a `i`, každý `range_inclusive(-r, r)`.
2. Do součtu přidejte `weight(i, j)` krát barvu souseda, `src[at - ivec2(i, j)].rgb`.
3. Alfu vraťte z původního pixelu.
</details>

<details><summary>🔑 Řešení</summary>

""" + answer(SOLUTION_CONV) + r"""
</details>
""")

code(CONV, solve=[(CONV_STUB, SOLUTION_CONV)])

md(r"""
Přepínejte `preset`:

| `preset` | Co dělá |
|---|---|
| `Box 15 x 15` | průměr čtverce, rozmazání ze sekce 1 |
| `Disc 25 x 25` | průměr kruhu, rozmazání jako objektivem |
| `Gauss 25 x 25` | měkké rozmazání, sekce 3 |
| `Motion 25 x 1` | průměr řádku, rozmazání pohybem do strany |
| `Sharpen 9 x 9` | k pixelu přičte dvakrát jeho rozdíl od průměru okolí, zaostří |
| `Edges 5 x 5` | rozdíl pixelu od průměru okolí, zesílený 25krát, plochy zčernají |
| `Emboss 5 x 5` | hrany světlé z jedné strany a tmavé z druhé, reliéf |

Do mřížky jádra `Own 3 x 3` zkuste průměr `1 1 / 1 1`, hrany `-1 -1 / -1 8` a rozmazání jen do strany `0 0 / 1 1`. Na fotce 1200 × 800 je jádro 3 × 3 slabé, proto jsou hotová jádra větší.

@img(kernel_gallery.png, 900, Výřez fotky po konvoluci s pěti hotovými jádry: kruh, pohyb, zaostření, hrany a reliéf)

<details><summary>❓ Otázka: co říká součet vah?</summary>

Součet vah průměru i zaostření je 1, součet vah hran je 0. Co součet vah říká o tom, co jádro udělá s jednobarevnou plochou?

<details><summary>🔑 Odpověď</summary>

Na ploše jsou všichni sousedé stejní, takže výsledek je barva plochy krát součet vah. Se součtem 1 plocha zůstane, jak byla, a jádro mění jen místa, kde se sousedé liší. Se součtem 0 plocha zčerná a zbydou jen změny, hrany. Proto `weight` dělí váhy jejich součtem, a když je součet 0, nechá je být.
</details>

</details>

Konvoluce zatím vážila čísla sRGB, ta ale nejsou množství světla. **Rozmazání proto počítáme v lineárním RGB**, které míchá světlo jako objektiv. **Zaostření a hrany počítáme ze světlosti L v OKLabu**, která měří rozdíly, jak je vidí oko, a barvu nechá být.

<details><summary>📖 V jakém prostoru počítat</summary>

Na sytých barvách je rozdíl vidět nejlíp. Obrázek ukazuje slunce z pravého horního rohu fotky, rozmazané a zaostřené stejným průměrem okolí ve třech prostorech. Zaostření v OKLabu mění jen světlost L.

@img(spaces.png, 640, Žluté slunce na modré obloze rozmazané a zaostřené v sRGB, v lineárním RGB a v OKLabu)

Rozmazání v sRGB udělá mezi žlutou a modrou tmavý olivový pruh, protože průměr čísel sRGB je tmavší než průměr světla. Lineární RGB smíchá žlutou s modrou jako objektiv, do světlého přechodu. OKLab vyjde mezi nimi.

Zaostření přičte k pixelu jeho rozdíl od průměru okolí. V sRGB i v lineárním RGB se to děje v každém kanálu zvlášť, a lem kolem slunce je proto uvnitř sytě žlutý a venku sytě modrý: zaostření změnilo i barvu. Zaostření světlosti L v OKLabu udělá lem uvnitř světlejší a venku tmavší a barvu nechá. Je to vidět po rozkladu okraje na světlost L a sytost C, vzdálenost barvy od šedé:

@img(spaces_split.png, 640, Okraj slunce před zaostřením, po zaostření v sRGB a po zaostření světlosti v OKLabu, rozložený na světlost L a sytost C)

Zaostření v sRGB zvedlo sytost v pruhu podél okraje, zaostření L v OKLabu ji nechalo skoro stejnou a změnilo jen světlost. Tenká tmavá čára přímo na okraji je už v původním obrázku: pixel na rozhraní je směs žluté a modré, a ta je skoro šedá.

Chcete to vidět v Sáře? Vložte kernel do nové buňky, přepínejte `space` a dívejte se na slunce vpravo nahoře:

""" + answer(SPACES, "") + r"""
</details>
""")

# --- 3. Gaussian and separability ----------------------------------------------------
md(r"""
## 3. Gauss ve dvou průchodech

Průměr okolí udělá z jasného bodu hranatou skvrnu, protože soused v rohu čtverce váží stejně jako ten vedle. **Gaussovo rozmazání** váží souseda podle vzdálenosti a váha s ní plynule klesá. Podél řádku je to Gaussova křivka se šířkou $\sigma$:

$$k(i) = e^{-\frac{i^2}{2\sigma^2}}$$

Posuňte `sigma`. Ve vzdálenosti $3\sigma$ je váha asi setina té prostřední, kernel proto sčítá jen do poloměru $r = \lceil 3\sigma \rceil$.
""")

code(GAUSS_1D_PLOT, hidden=True)

md(r"""
Ve dvou rozměrech váha závisí na vzdálenosti od pixelu uprostřed, $\sqrt{i^2 + j^2}$. Čáry stejné váhy jsou proto kružnice a jasný bod se rozmaže do kulaté skvrny:

$$w(i, j) = e^{-\frac{i^2 + j^2}{2\sigma^2}}$$
""")

code(GAUSS_2D_PLOT, hidden=True)

md(r"""
Kernel sčítá sousedy do poloměru $r$ s vahami $w(i, j)$ a součet dělí součtem vah. Počítá v lineárním RGB. Poloměr roste se `sigma` až na 24 pixelů, a kernel proto má na prvním řádku `halo=32`: smí číst až 32 pixelů za okrajem vrstvy. Bez něj Sára čte jen 16 pixelů za okraj a dál dostane průhlednou černou.
""")

code(GAUSS_2D)

md(r"""
Gauss rozmaže světla do měkkých kulatých skvrn, blíž objektivu než průměr čtverce. Objektiv ale dělá kruhy s ostrým okrajem, stejně jasné od středu až k okraji. Gauss má jas nejvyšší uprostřed a k okraji ho plynule ztrácí:

@img(blur_shapes.png, 900, Pravá část fotky rozmazaná průměrem čtverce se čtvercovými světly, Gaussem s měkkými kulatými a objektivem s kulatými s ostrým okrajem)

<details><summary>❓ Otázka: proč Gauss nepočítá v OKLabu?</summary>

OKLab měří rozdíly tak, jak je vidí oko. Proč v něm rozmazání nepočítáme?

<details><summary>🔑 Odpověď</summary>

Rozmazání napodobuje objektiv a ten sčítá světlo. Množství světla je lineární RGB. OKLab bere ze světla třetí odmocninu, a jasné body proto v průměru váží málo: odlesky se po rozmazání zmenší a ztmavnou. Na slunci v sekci 2 je to vidět. OKLab se hodí tam, kde se měří rozdíl, který vidí oko: u zaostření, hran a barevné váhy bilaterálního filtru.
</details>

</details>
""")

md(r"""
### Oddělitelnost

Kernel čte $(2r + 1)^2$ sousedů, při $\sigma = 8$ je to 2401 sousedů na pixel. Gaussova váha je ale součin váhy podle $i$ a váhy podle $j$:

$$e^{-\frac{i^2 + j^2}{2\sigma^2}} = e^{-\frac{j^2}{2\sigma^2}} \cdot e^{-\frac{i^2}{2\sigma^2}}$$

Konvoluce řádku $g_\text{ř}$ se sloupcem $g_\text{s}$ dá právě matici těchto součinů, takže Gaussovo jádro je $G = g_\text{ř} * g_\text{s}$. Konvoluce je navíc asociativní, a proto

$$f * G = f * (g_\text{ř} * g_\text{s}) = (f * g_\text{ř}) * g_\text{s}$$

Stačí rozmazat každý řádek a výsledek pak rozmazat po sloupcích. Dva průchody čtou $2(2r + 1)$ sousedů místo $(2r + 1)^2$, při $\sigma = 8$ tedy 98 místo 2401. Jádro, které jde takto rozložit na sloupec a řádek, je **oddělitelné** (*separable*).

@img(separable.png, 900, Dvourozměrné Gaussovo jádro jako součin sloupce a řádku a počet čtených sousedů obou postupů v závislosti na poloměru)

<details><summary>📖 Proč je konvoluce asociativní</summary>

Je to součet součinů a ten jde sčítat v libovolném pořadí (posuny $\mathbf{u}, \mathbf{v}, \mathbf{t}$ jsou dvourozměrné):

$$\bigl((f * a) * b\bigr)(\mathbf{x}) = \sum_{\mathbf{v}} b(\mathbf{v}) \sum_{\mathbf{u}} a(\mathbf{u})\, f(\mathbf{x} - \mathbf{u} - \mathbf{v}) = \sum_{\mathbf{t}} \underbrace{\Bigl(\sum_{\mathbf{u} + \mathbf{v} = \mathbf{t}} a(\mathbf{u})\, b(\mathbf{v})\Bigr)}_{(a * b)(\mathbf{t})} f(\mathbf{x} - \mathbf{t}) = \bigl(f * (a * b)\bigr)(\mathbf{x})$$
</details>

<details><summary>❓ Otázka: je oddělitelný průměr a jádro hran?</summary>

Je oddělitelný průměr 3 × 3? A jádro hran −1 −1 −1 / −1 8 −1 / −1 −1 −1?

<details><summary>🔑 Odpověď</summary>

Průměr ano, je to sloupec $(1\;1\;1)^{\mathsf T} / 3$ s řádkem $(1\;1\;1) / 3$. Jádro hran ne. V matici sloupce s řádkem je každý řádek násobkem téhož řádku, ale $(-1\;\,{-1}\;\,{-1})$ a $(-1\;\,8\;\,{-1})$ násobky nejsou.
</details>

</details>

### 🎯 Úkol 2: první průchod

Napište první průchod, rozmazání po řádku. Pro řádek $y$ a $k(i) = e^{-i^2 / 2\sigma^2}$ je

$$g(x, y) = \frac{\sum_{i=-r}^{r} k(i)\, f(x + i, y)}{\sum_{i=-r}^{r} k(i)}$$

Výsledek nechte v lineárním RGB, druhý průchod ho čte tak, jak je, a do sRGB převede až výsledek. V Sáře proto vrstva vypadá tmavší. Nedokončený kernel jen převede pixel do lineárního RGB.

<details><summary>💡 Nápověda</summary>

1. Vezměte kernel `Gauss 2D` a nechte jen vnitřní cyklus, přes `dx`.
2. Váha závisí jen na `dx`.
3. Na konci vraťte součet dělený součtem vah.
</details>

<details><summary>🔑 Řešení</summary>

""" + answer(SOLUTION_BLUR_X) + r"""
</details>
""")

code(BLUR_X, solve=[(BLUR_X_STUB, SOLUTION_BLUR_X)])

md(r"""
Druhý průchod je hotový a čte vrstvu `Blur X`. Po změně `sigma` v prvním průchodu nastavte stejnou hodnotu i ve druhém, sám se nespustí. Vrstva `Blur Y` má vypadat stejně jako `Gauss 2D`.
""")

code(BLUR_Y)

md(r"""
<details><summary>📖 Kolik to ušetří</summary>

Kolik času dva průchody ušetří, změří `sara.bench`. Nechte všude `sigma` na 3 a spusťte v nové buňce:

```python
two_d = sara.bench("Gauss 2D", runs=10)
rows = sara.bench("Blur X", runs=10)
columns = sara.bench("Blur Y", runs=10)
print(f"one pass, 2D:  {two_d.gpu_ms:8.2f} ms")
print(f"two passes:    {rows.gpu_ms + columns.gpu_ms:8.2f} ms")
```

Nastavte `sigma` ve všech třech kernelech na 8 a buňku spusťte znovu. Poloměr vzrostl z 9 na 24. Jeden průchod čte $49^2 / 19^2$, asi 6.7krát víc sousedů, dva průchody $98 / 38$, asi 2.6krát víc. Naměřené poměry tomu odpovídají jen zhruba. GPU čte sousedy z rychlé vyrovnávací paměti (*cache*), a u malého poloměru tvoří většinu času spuštění a zápis vrstvy. Dva průchody navíc celou vrstvu `Blur X` jednou zapíšou a znovu přečtou, takže při malém poloměru nemusí vyhrát. Nakonec vraťte `sigma` všude na 3.
</details>
""")

# --- 4. Sobel ---------------------------------------------------------------------
md(r"""
## 4. Hrany

Hrana je místo, kde se rychle mění světlost. Jak rychle se mění, měří derivace. Pro světlost $f$ podél řádku je to sklon sečny přes krok $h$, když se krok zmenšuje k nule:

$$f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}$$

Pixely ale mají krok nejméně jeden pixel, menší $h$ nemáme. Limitu proto vynecháme a dosadíme $h = 1$. Vyjde **dopředná diference**:

$$f'(x) \approx \frac{f(x + 1) - f(x)}{1} = f(x + 1) - f(x)$$

Je to jádro $(-1\;\,1)$ na pixelech $x$ a $x + 1$. Měří ale sklon mezi nimi, o půl pixelu vpravo od $x$. Zpětná diference $f(x) - f(x - 1)$ měří sklon o půl pixelu vlevo. Jejich průměr leží přesně na $x$ a je to **centrální diference**:

$$f'(x) \approx \frac{\big(f(x + 1) - f(x)\big) + \big(f(x) - f(x - 1)\big)}{2} = \frac{f(x + 1) - f(x - 1)}{2}$$

Totéž dá limita se sečnou souměrnou kolem $x$, $f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x - h)}{2h}$, s $h = 1$. Váhy sousedů $x - 1$, $x$, $x + 1$ jsou $\frac{1}{2}(-1\;\,0\;\,1)$. Polovinu vynecháme, hranu najde i dvojnásobek a sílu stejně nastaví posuvník `gain`. Zbude jádro $(-1\;\,0\;\,1)$: pravý soused minus levý.

<details><summary>📖 Proč je centrální diference přesnější</summary>

Taylorův rozvoj pro souseda ve vzdálenosti $h$ na obě strany:

$$f(x \pm h) = f(x) \pm h\,f'(x) + \frac{h^2}{2}\,f''(x) \pm \frac{h^3}{6}\,f'''(x) + O(h^4)$$

Dopředná diference z něj dá $\frac{f(x + h) - f(x)}{h} = f'(x) + \frac{h}{2}\,f''(x) + O(h^2)$, chybu úměrnou $h$. Po odečtení obou řádků se členy se sudou mocninou $h$ odečtou, a centrální diference dá $\frac{f(x + h) - f(x - h)}{2h} = f'(x) + \frac{h^2}{6}\,f'''(x) + O(h^4)$, chybu úměrnou $h^2$. Na pixelech je $h = 1$ a chyba závisí na tom, jak rychle se fotka mění. Na pomalých změnách je centrální diference přesnější a hlavně nemá posun o půl pixelu.
</details>

Ve 2D má světlost $f(x, y)$ dvě **parciální derivace**, $\frac{\partial f}{\partial x}$ podél řádku a $\frac{\partial f}{\partial y}$ podél sloupce. Obě se spočítají stejně: jádro $(-1\;\,0\;\,1)$ v řádku a totéž jádro postavené do sloupce. Jeden řádek ale zachytí i šum. **Sobelův operátor** proto rozdíl zprůměruje přes tři řádky s vahami $(1\;\,2\;\,1)$. Podle sekce 3 je to konvoluce sloupce s řádkem, oddělitelné jádro:

$$G_x = \begin{pmatrix} 1 \\ 2 \\ 1 \end{pmatrix} * \begin{pmatrix} -1 & 0 & 1 \end{pmatrix} = \begin{pmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{pmatrix} \qquad G_y = G_x^{\mathsf T}$$

Dvojice derivací $(g_x, g_y)$ je **gradient**. Jeho délka je síla hrany, jeho směr míří od tmavé strany ke světlé, kolmo na hranu.

@img(sobel.png, 900, Sobelova jádra Gx a Gy, gradient na hraně kolmo na ni a barevné kolo směrů, které kernel používá)

Kernel níže je hotový a počítá gradient světlosti OKLab L, $L(x, y)$ je `light(p)`:

$$g_x = \sum_{j=-1}^{1} \sum_{i=-1}^{1} G_x(i, j)\, L(x + i, y + j), \qquad g_y = \sum_{j=-1}^{1} \sum_{i=-1}^{1} G_y(i, j)\, L(x + i, y + j)$$

`g.x` je součet s jádrem $G_x$, `g.y` s jádrem $G_y$, oba najednou v jednom cyklu přes okolí. Váha $G_x$ v políčku `(i, j)` je `i * (2 - abs(j))`, váha $G_y$ totéž s prohozenými `i` a `j`. Funkce `shade` obarví hranu podle směru, když zapnete `direction`. Zkuste `gain` a `direction`.
""")

code(SOBEL)

md(r"""
<details><summary>❓ Otázka: proč mají dvě strany čáry opačné barvy?</summary>

Zapněte `direction`. Proč mají dvě strany tenké čáry, třeba šikmé čáry vpravo, opačné barvy?

<details><summary>🔑 Odpověď</summary>

Barva je směr gradientu, od tmavší strany ke světlejší. Tenká černá čára má světlé okolí na obou stranách: přes jednu její stranu světlost klesá, přes druhou roste, a gradienty obou stran míří od čáry, proti sobě. Opačné směry dostanou na barevném kole opačné barvy. Cannyho detektor hran to využívá: hranu ztenčí na jeden pixel tak, že nechá jen místa, kde je síla největší ve směru gradientu.
</details>

</details>
""")

# --- 5. NumPy and Krita, at home -----------------------------------------------------------
md(r"""
## 5. Doma: do Krity

Krita kernely nespouští a NumPy nemá vlákno na pixel, umí ale celé pole najednou. Posunutý obrázek je **výřez** rozšířeného pole, `p[:, i:i + w]`, pohled na tatáž data bez kopie. Vážený součet $2r + 1$ výřezů je jeden průchod rozmazání.

@img(numpy_shifts.png, 760, Řádek rozšířený o okraje a pět výřezů posunutých o jeden pixel, každý vynásobený svou vahou a sečtený)

Filtr je níže přepsaný do NumPy hotový, doma ho spustíte a vložíte do Krity. Spusťte obě buňky s kódem a pak třetí. První obsahuje převody barev, váhy Gaussova jádra `gaussian_1d` a `PARAMS`, druhá `blur` a `apply`.

**Zaostření** (*unsharp mask*) přičte k obrázku jeho rozdíl od rozmazaného obrázku. Rozdíl jsou detaily menší než $\sigma$ a `sharpen` $a$ říká, kolikrát je zesílit:

$$g = f + a\,\bigl(f - \text{blur}(f)\bigr)$$

`Blur` počítá v lineárním RGB, `Sharpen` zaostří jen světlost L v OKLabu, jak ukázala sekce 2.
""")

code(HELPERS_NP + "\n\n\n" + PARAMS_NP)

code(SOLUTION_BLUR_NP + "\n\n\n" + SOLUTION_APPLY)

code("""
sara.live(apply, PARAMS, source="coffee", target="Filter NumPy")
""")

md(r"""
Vrstva `Filter NumPy` vypadá stejně jako `Goal`.

<details><summary>📖 Co se v NumPy mění oproti kernelu</summary>

Kernel počítá jeden pixel, NumPy celé pole najednou. Tím se liší hlavně tyhle řádky:

| Kernel | NumPy |
|---|---|
| `pixel(ivec2 at)` pro jeden pixel | `apply(img, ...)` pro celé pole |
| cyklus přes `dx` sčítá $2r + 1$ sousedů | `sum(k[i] * p[:, i:i + w] for i in range(len(k)))`, výřezy celého pole |
| Sára čte za okrajem nejbližší pixel | `np.pad(a, ..., mode="edge")`, Clamp |
| `srgb_to_linear_srgb`, `linear_srgb_to_oklab` z knihoven | `srgb_to_linear` a `linear_to_oklab` v pomocných funkcích, s maticemi |
| `halo` a `sigma` jako parametry kernelu | `PARAMS` a argumenty `apply` |

</details>

### 🎯 Úkol 3: filtr v Kritě

V `krita/pga_filter/effect.py` nahraďte všechno pod řádkem `import numpy as np` obsahem obou buněk s kódem výše a nastavte `TITLE = "Blur and sharpen"`. Vyzkoušejte filtr v Kritě na vlastní fotce a porovnejte rychlost s Gaussovým rozmazáním, které má Krita vestavěné.

<details><summary>💡 Nápověda</summary>

1. Do `effect.py` patří pomocné funkce, `PARAMS`, `blur` a `apply`, nic z buňky se `sara.live`.
2. Náhled počítá jen malou kopii, pomalé je až Apply na celé fotce. Zkuste filtr nejdřív na výběru. Ve zmenšeném náhledu „Whole region, scaled down“ vypadá rozmazání silnější, skutečnou sílu ukáže „Centre at full size“.
</details>

<details><summary>❓ Otázka: proč má vystřižený předmět po rozmazání tmavý lem?</summary>

Vystřižený předmět na průhledné vrstvě má po rozmazání kolem sebe tmavý lem. Proč, a jak mu předejít?

<details><summary>🔑 Odpověď</summary>

Průhledné pixely mají nějakou barvu, obvykle černou, a rozmazání ji přimíchá k barvě předmětu, i když ji nikdo nevidí. Pomůže rozmazávat barvu vynásobenou alfou (*premultiplied alpha*), rozmazat i alfu a výslednou barvu alfou zase vydělit. Průhledný pixel pak k barvě nepřidá nic.
</details>

</details>
""")

# --- 6. Bilateral, a bonus -----------------------------------------------------------------
md(r"""
## 6. Rozmazání, které nechá hrany (nepovinné)

Tohle je bonus, do hodiny se nevejde.

Gauss vyhladí šum a drobnou texturu, ale stejně rozmaže i hrany, třeba okraj hrnku. Chceme vyhladit plochy a hrany nechat ostré. Soused za hranou má jinou barvu než pixel uprostřed, dáme mu tedy malou váhu. **Bilaterální filtr** (*bilateral filter*) násobí Gaussovu váhu podle vzdálenosti druhou Gaussovou vahou podle rozdílu barev $\mathbf{c}$ v OKLabu:

$$w(i, j) = e^{-\frac{i^2 + j^2}{2\sigma^2}} \cdot e^{-\frac{\lVert \mathbf{c}(x + i,\, y + j) - \mathbf{c}(x, y) \rVert^2}{2\sigma_c^2}}$$

Výsledek se jako u Gausse dělí součtem vah.

@img(bilateral.png, 900, Okolí pixelu na okraji hrnku, váha podle vzdálenosti, váha podle barvy a jejich součin, který za hranou skoro mizí)

### 🎯 Bonus: bilaterální filtr

Doplňte barevnou váhu, druhý činitel vzorce výše. Funkce `lab(p)` vrátí barvu pixelu `p` v OKLabu, `centre` je barva pixelu uprostřed a `sigma_c` je $\sigma_c$. Nedokončený kernel je Gauss 2D.

<details><summary>💡 Nápověda</summary>

1. Rozdíl barev je vektor `lab(p) - centre`, čtverec jeho délky dá `dot`.
2. Váhu `k` vynásobte druhým `exp(...)`, stejně utvořeným jako ten prostorový.
</details>

<details><summary>🔑 Řešení</summary>

""" + answer(SOLUTION_BILATERAL.replace("            ", "")) + r"""
</details>
""")

code(BILATERAL, solve=[(BILATERAL_STUB, SOLUTION_BILATERAL)])

md(r"""
Porovnejte vrstvy `Bilateral` a `Gauss 2D`. S `sigma_c` 0.05 zůstanou okraj hrnku i lžička ostré, podšálek a dřevo se vyhladí. S velkou `sigma_c` se výsledek blíží `Gauss 2D`.

<details><summary>❓ Otázka: proč to nejde ve dvou průchodech?</summary>

Proč bilaterální filtr nejde spočítat ve dvou průchodech jako Gauss?

<details><summary>🔑 Odpověď</summary>

Barevná váha závisí na barvě pixelu uprostřed, a každý pixel má proto jiné jádro. Filtr není konvolucí s jedním jádrem, a tak na něj asociativita nejde použít. Dva průchody po řádcích a sloupcích se používají jen jako rychlá náhrada s chybami u šikmých hran.
</details>

</details>
""")

# --- 7. Edges of the picture -----------------------------------------------------------
md(r"""
## 7. Za okrajem obrázku (nepovinné)

Pixel u okraje nemá sousedy na všech stranách. Co číst za okrajem, je potřeba zvolit:

| Pravidlo | Za okrajem je | V NumPy |
|---|---|---|
| Zero | černá | `np.pad(..., mode="constant")` |
| Clamp | nejbližší pixel na okraji, okraj se natáhne | `mode="edge"` |
| Mirror | obrázek zrcadlený podle okraje | `mode="symmetric"` |
| Wrap | protější okraj, obrázek se opakuje jako dlaždice | `mode="wrap"` |

@img(edge_modes.png, 900, Fotka posunutá doprava dolů a čtyři pravidla pro to, co se objeví za okrajem: černá, natažený okraj, zrcadlo a dlaždice)

Sára sama čte za okrajem nejbližší pixel, tedy Clamp, až do vzdálenosti `halo`. Kernel níže je hotový, posune fotku o `dx` a `dy` a za okrajem čte podle pravidla `mode`. Rozmazání u okraje čte za okraj stejně, jen méně daleko.

<details><summary>📖 Jak fungují Mirror a Wrap</summary>

Clamp je `clamp` po složkách. Mirror pro souřadnici $x$ v plátně šířky $W$ zrcadlí záporné $x$ na $-1 - x$ a $x \ge W$ na $2W - 1 - x$, což jde napsat funkcemi `max` a `min`. Wrap je zbytek po dělení šířkou, který je i pro záporné $x$ od 0 do $W - 1$. Operátor `%` v GLSL na záporných číslech výsledek nezaručuje, proto kernel počítá $x - W \lfloor x / W \rfloor$ s dělením ve `float`.
</details>
""")

code(EDGES)

md(r"""
<details><summary>❓ Otázka: které pravidlo se k čemu hodí?</summary>

Které pravidlo se hodí k rozmazání fotky, které k rozmazání dlaždice textury a proč je Zero u rozmazání skoro vždy špatně?

<details><summary>🔑 Odpověď</summary>

U fotky Mirror nebo Clamp: za okrajem pokračuje něco podobného tomu, co je u okraje. Clamp u velkého poloměru dělá pruhy z natažených pixelů, Mirror ne. Textura, která se má opakovat, potřebuje Wrap, jinak by se na švech dlaždic objevil skok. Zero přimíchá do okrajů černou a okraje rozmazané fotky ztmavnou. Fourierova transformace v lekci o frekvencích bere obrázek vždy jako dlaždici, tedy Wrap.
</details>

</details>
""")

# --- Optional -----------------------------------------------------------------------
md(r"""
## Nepovinné: sdílená paměť a druhá derivace

Obojí je rozšíření, které se do hodiny nevejde.

<details><summary>📖 Sdílená paměť: rychlejší první průchod</summary>

V prvním průchodu čte každé vlákno $2r + 1$ pixelů řádku a sousední vlákna čtou skoro tytéž pixely. Při $\sigma = 8$ přečte GPU každý pixel 49krát a 49krát ho převede ze sRGB. Vlákna jedné **skupiny** (*workgroup*) mají společnou rychlou **sdílenou paměť** (*shared memory*). Skupina si do ní řádek načte jednou, každé vlákno jeden nebo dva pixely, počká na ostatní a pak sčítá ze sdílené paměti.

@img(shared_tile.png, 900, Skupina 64 vláken načte do sdílené paměti svých 64 pixelů a 24 pixelů na každé straně, pak každé vlákno sečte 49 hodnot ze sdílené paměti)

Funkce `pixel` sdílenou paměť nemá, kernel je proto napsaný celý: řádek `kernel` se jménem, obrázky `src` a `dst`, parametry pod `params:` a skupina 64 vláken v řádku. `local_id` je číslo vlákna ve skupině, `group_id` číslo skupiny a `pixel` pozice vlákna v obrázku. `barrier()` počká, až všechna vlákna skupiny dojdou na totéž místo, jinak by vlákno mohlo číst, kam soused ještě nezapsal. Parametr celého kernelu nemá výchozí hodnotu, dává mu ji první řádek buňky, `sigma=3`. Celý kernel vidí obrázek včetně okraje `halo`, takže jeho `pixel` je posunutý o 32 proti pozici na plátně. Zápis do `dst` na stejný `pixel` přistane na správném místě.

Vložte kernel do nové buňky a v další buňce změřte oba první průchody:

""" + answer(SHARED, "") + r"""

```python
shared = sara.bench("Blur X shared", runs=10)
plain = sara.bench("Blur X", runs=10)
print(f"first pass, plain:   {plain.gpu_ms:8.3f} ms")
print(f"first pass, shared:  {shared.gpu_ms:8.3f} ms")
```

Zkuste měření s `sigma` 1, 3 a 8 v obou kernelech. Sdílená paměť se vyplatí, když vlákna čtou hodně stejných pixelů a čtení je drahé: velký poloměr a převod ze sRGB u každého čtení. Kernel se sdílenou pamětí čte a převádí každý pixel nanejvýš dvakrát, ať je poloměr jakýkoli, a jeho čas skoro neroste. Při malém poloměru je rozdíl malý, protože GPU drží nedávno čtené pixely ve vyrovnávací paměti samo, a načtení do sdílené paměti a čekání na `barrier()` něco stojí. Velikost sdíleného pole musí být známá při překladu, a poloměr je proto omezený konstantou `R_MAX`.
</details>

<details><summary>📖 Druhá derivace a Laplaceův operátor</summary>

Sobel měří první derivaci, strmost. Hrana je tam, kde je strmost největší, a v tom místě má **druhá derivace** nulu a mění znaménko: na tmavé straně hrany je kladná, na světlé záporná.

@img(second_derivative.png, 900, Měkká hrana vzorkovaná po pixelech, její první derivace s vrcholem na hraně, druhá derivace s průchodem nulou na hraně a hrana minus druhá derivace se zákmity na obou stranách)

Jádro pro druhou derivaci dá Taylorův rozvoj. Pro souseda ve vzdálenosti $h$ na obě strany:

$$f(x \pm h) = f(x) \pm h\,f'(x) + \frac{h^2}{2}\,f''(x) \pm \frac{h^3}{6}\,f'''(x) + O(h^4)$$

Po sečtení obou řádků se členy s lichou mocninou $h$ odečtou:

$$f(x + h) + f(x - h) = 2 f(x) + h^2 f''(x) + O(h^4)$$

$$f''(x) = \frac{f(x + h) - 2 f(x) + f(x - h)}{h^2} + O(h^2)$$

S krokem jednoho pixelu, $h = 1$, je to jádro $(1\;\,{-2}\;\,1)$. Ve 2D se druhé derivace v $x$ a v $y$ sečtou do **Laplaceova operátoru** $\nabla^2 f = f_{xx} + f_{yy}$, a sečtou se i jejich jádra:

$$\nabla^2 \approx \begin{pmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{pmatrix}$$

Jádro hran `-1 -1 / -1 8` z mřížky v úkolu 1 je záporný Laplace, který bere i sousedy na úhlopříčkách. Výchozí jádro mřížky, zaostření, je identita minus Laplace:

$$\begin{pmatrix} 0 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 0 \end{pmatrix} - \begin{pmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{pmatrix} = \begin{pmatrix} 0 & -1 & 0 \\ -1 & 5 & -1 \\ 0 & -1 & 0 \end{pmatrix}$$

Odečtení druhé derivace hranu zostří: na tmavé straně je $f''$ kladná a pixel ztmavne, na světlé je záporná a pixel zesvětlí. Na obrázku vpravo je to zákmit na obou stranách hrany.

Druhá derivace ale zesílí nejvíc ty nejjemnější detaily. Vlna $\cos \omega x$ vyjde z jádra $(1\;\,{-2}\;\,1)$ jako $-4\sin^2(\omega / 2)\,\cos\omega x$, tedy vynásobená číslem, které roste s frekvencí. Nejvíc tedy zesílí šum, a proto se Laplace počítá z rozmazaného obrázku (*Laplacian of Gaussian*, LoG).

Zaostření z části Doma: do Krity (*unsharp mask*) je také identita minus Laplace, jen ve větším měřítku. Taylorův rozvoj rozmazání do druhého řádu, zprůměrovaný s Gaussovými vahami, dá

$$\text{blur}(f) \approx f + \frac{\sigma^2}{2}\,\nabla^2 f, \qquad f + a\,(f - \text{blur}(f)) \approx f - a\,\frac{\sigma^2}{2}\,\nabla^2 f$$

Zvolte v úkolu 1 jádro `Own 3 x 3` a zadejte do mřížky Laplaceovo jádro, `0 1 / 1 -4`. Vrstva je skoro celá černá, protože součet vah je 0 a každá plocha zčerná. Svítí jen tmavá strana hran, kde je druhá derivace kladná, a jemná textura. Na světlé straně hran je výsledek záporný a obrazovka ho ukáže jako černou. Nakonec vraťte do mřížky zaostření, `0 -1 / -1 5`.
</details>
""")

# --- Summary ------------------------------------------------------------------
md(r"""
## Shrnutí

- Rozmazání míchá pixel s okolím: průměr celé fotky je jedna barva, průměr po blocích mozaika, průměr okolí každého pixelu rozmazání. Objektiv rozlije bod do kruhu, průměr okolí do čtverce.
- Konvoluce je vážený součet okolí. Součet vah 1 zachová plochy, součet 0 nechá jen změny.
- Konvoluce je asociativní, a jádro, které je konvolucí sloupce s řádkem, jde proto spočítat ve dvou průchodech: $2(2r + 1)$ sousedů místo $(2r + 1)^2$. Takové je Gaussovo jádro, průměr i Sobel.
- Rozmazání míchá světlo, a proto počítá v lineárním RGB. Zaostření a hrany měří rozdíly, jak je vidí oko, a proto počítají ze světlosti L v OKLabu.
- Sobelův operátor odhadne gradient světlosti. Jeho délka je síla hrany, směr míří kolmo na hranu.
- NumPy počítá konvoluci jako vážený součet posunutých výřezů celého pole, s okrajem podle zvoleného pravidla.
- Bilaterální filtr (nepovinný) váží souseda i podle rozdílu barev a hrany nerozmaže. Jádro se mění od pixelu k pixelu, a dva průchody proto nejdou.
- Co bude dál: konvoluce v obraze je násobení ve spektru. Právě to vysvětlí lekce 4, která následuje hned po této.

<details><summary>📖 Co jsme vynechali</summary>

- **Rozmazání s proměnným poloměrem.** Hloubka ostrosti nebo rozmazání podle masky mění $\sigma$ pixel od pixelu a jádro pak není oddělitelné.
- **Rychlé aproximace.** Několik průměrů za sebou se blíží Gaussovi. Kawaseho rozmazání a pyramida zmenšených obrázků jsou rychlé způsoby, jak hry dělají rozmazání pro bloom.
- **Medián.** Prostřední hodnota okolí místo průměru hrany drží, ale na GPU se musí třídit.
- **Cannyho detektor hran.** Gaussovo rozmazání, Sobel, ztenčení hran na jeden pixel a dvě prahové hodnoty dají dohromady čisté čáry místo šedých pruhů.
- **Konvoluce přes Fourierovu transformaci.** Pro velká jádra je rychlejší násobit spektra, jak ukáže lekce o frekvencích.

</details>

<details><summary>🎯 Bonusové úkoly</summary>

1. **Průměr v konstantním čase.** Napište průměr okolí v NumPy jako rozdíl kumulativních součtů, `np.cumsum`, takže čas nezávisí na poloměru. Třikrát za sebou dá skoro Gausse.
2. **Bilaterální filtr v NumPy.** Přidejte do filtru pro Kritu režim `Bilateral`. Jako u `blur` je to vážený součet výřezů, jen váha každého výřezu je pole, ne číslo.
3. **Ostření jen na hranách.** Použijte sílu hrany ze Sobela jako masku zaostření, aby se nezesílil šum na plochách.
4. **Pravidla okraje sami.** Smažte v kernelu `Shifted` větve `CLAMP`, `MIRROR` a `WRAP` a napište je znovu, každá z pozice `p` spočítá pozici uvnitř plátna. Nápověda je v 📖 pod kernelem.

</details>

<details><summary>📖 Nápady na semestrální práci</summary>

- Rozostření podle masky: plugin rozmaže vrstvu podle světlosti druhé vrstvy nebo výběru, jako tilt-shift fotografie.
- Detektor hran s volbou operátoru (Sobel, Scharr, Prewitt) a výstupem jako čárová kresba nebo maska výběru.
- Vyhlazení pleti nebo šumu bilaterálním filtrem s maskou, kde se vyhlazuje.

</details>
""")


def build():
    lesson_writer.write(cells, OUT, "l3")


if __name__ == "__main__":
    build()
