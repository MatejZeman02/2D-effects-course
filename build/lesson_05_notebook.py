"""Builds the lesson 5 notebook in the lecture folder above this one.

The notebook is generated: edit this file and run it again, never the .ipynb
by hand, or the next build overwrites the edit. The pictures and the test photo
chelsea.png come from lesson_05_diagrams.py, which enlarges scikit-image's
cat photo (CC0) twice, as imgs/5/SOURCES.md says.

Markdown cells are raw strings, so LaTeX keeps single backslashes and braces.
A picture from imgs/5 is written as @img(file, width, alt text).

The solutions are constants near the top, so the markdown shows exactly the
code that test_lesson_05.py runs.
"""
import re
import textwrap
from pathlib import Path

import lesson_writer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "lesson_05_stylisation.ipynb"
IMG = "imgs/5"
WIDTH, HEIGHT = 902, 600      # the photo's size and the canvas's
SIGMA, K = 1.5, 1.6           # the two blurs, sigma and k sigma
P, EPSILON, PHI = 20.0, 0.5, 10.0
LEVELS = 5
RADIUS = 4                    # Kuwahara's opening radius
SPACING, ANGLE, THICK, LIGHTEST = 6.0, 45.0, 0.35, 0.75     # the hatching's opening values
BAND = 0.17                   # the lightness a hatching layer needs to go from nothing to full width
CELL = 9.0                    # the halftone's opening cell, in pixels
FLOW_SIGMA, REACH = 3.0, 4.0  # the flow's neighbourhood and the flow DoG's stroke length
SKETCH_SIGMA, SKETCH_GAIN = 4.0, 3.0
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
BLURS = f"""\
%%gmacs "chelsea" -> "Blurs" halo=32
import linear_srgb_color_space
import oklab_color_space

uniform float sigma: hint_range(0.3, 4) = {SIGMA}
uniform float k: hint_range(1.1, 3) = {K}


def light(ivec2 p) -> float:
    return linear_srgb_to_oklab(srgb_to_linear_srgb(src[p].rgb)).x


# Two Gaussian blurs of the lightness at once: sigma into red, k sigma into
# green, and the lightness itself into blue. Each blur stops at its own 3 sigma.
def pixel(ivec2 at) -> vec4:
    float wide = k * sigma
    int r1 = int(ceil(3.0 * sigma))
    int r2 = int(ceil(3.0 * wide))
    vec2 total = vec2(0.0)
    vec2 weight = vec2(0.0)
    for dy in range_inclusive(-r2, r2):
        for dx in range_inclusive(-r2, r2):
            float d2 = dx * dx + dy * dy
            vec2 w = exp(-d2 / (2.0 * vec2(sigma * sigma, wide * wide)))
            if abs(dx) > r1 or abs(dy) > r1:
                w.x = 0.0
            total += w * light(at + ivec2(dx, dy))
            weight += w
    vec2 g = total / weight
    return vec4(g.x, g.y, light(at), 1.0)"""

DOG_STUB = """\
    # TODO: the difference of the two blurs, times gain, around grey 0.5
    return vec4(vec3(b.b), 1.0)"""

SOLUTION_DOG = """\
    float d = b.r - b.g
    return vec4(vec3(0.5 + gain * d), 1.0)"""

DOG = f"""\
%%gmacs "Blurs" -> "DoG"
uniform float gain: hint_range(1, 50) = 10.0


def pixel(ivec2 at) -> vec4:
    vec4 b = src[at]
{DOG_STUB}"""

XDOG_STUB = """\
    # TODO: S sharpened by p, white from epsilon up and a soft tanh step below it
    return vec4(vec3(b.b), 1.0)"""

SOLUTION_XDOG = """\
    float s = (1.0 + p) * b.r - p * b.g
    float t = 1.0
    if s < epsilon:
        t = 1.0 + tanh(phi * (s - epsilon))
    return vec4(vec3(t), 1.0)"""

XDOG = f"""\
%%gmacs "Blurs" -> "XDoG"
uniform float p: hint_range(0, 100) = {P}
uniform float epsilon: hint_range(0, 1) = {EPSILON}
uniform float phi: hint_range(0, 100) = {PHI}


def pixel(ivec2 at) -> vec4:
    vec4 b = src[at]
{XDOG_STUB}"""

# Given finished, the toon kernel reads the photo and the XDoG layer.
TOON = f"""\
%%gmacs "chelsea" -> "Toon" lines=XDoG levels={LEVELS}
kernel toon

import linear_srgb_color_space
import oklab_color_space

image src: readonly
image lines: readonly
image dst

params:
    int levels: hint_range(2, 12)


def toon():
    vec3 lab = linear_srgb_to_oklab(srgb_to_linear_srgb(src[pixel].rgb))
    float ink = lines[pixel].r
    float n = levels - 1
    lab.x = floor(clamp(lab.x, 0.0, 1.0) * n + 0.5) / n
    lab *= ink
    vec3 rgb = clamp(oklab_to_linear_srgb(lab), vec3(0.0), vec3(1.0))
    dst[pixel] = vec4(linear_srgb_to_srgb(rgb), 1.0)"""

QUADRANT_STUB = """\
    # TODO: the mean colour of the (radius + 1)^2 pixels, and the variance of their luminance
    return vec4(srgb_to_linear_srgb(src[at].rgb), 1.0)"""

SOLUTION_QUADRANT = """\
    vec3 total = vec3(0.0)
    float s1 = 0.0
    float s2 = 0.0
    for dy in range_inclusive(0, radius):
        for dx in range_inclusive(0, radius):
            vec3 c = srgb_to_linear_srgb(src[at + towards * ivec2(dx, dy)].rgb)
            float l = dot(c, vec3(0.2126, 0.7152, 0.0722))
            total += c
            s1 += l
            s2 += l * l
    float n = (radius + 1) * (radius + 1)
    return vec4(total / n, s2 / n - (s1 / n) * (s1 / n))"""

KUWAHARA = f"""\
%%gmacs "chelsea" -> "Kuwahara"
import linear_srgb_color_space

uniform int radius: hint_range(1, 8) = {RADIUS}


def quadrant(ivec2 at, ivec2 towards) -> vec4:
    \"\"\"The square of side radius + 1 from at towards one corner: its mean colour in linear
    light, and in .a the variance of its luminance.\"\"\"
{QUADRANT_STUB}


def calmer(vec4 best, vec4 q) -> vec4:
    if q.a < best.a:
        return q
    return best


def pixel(ivec2 at) -> vec4:
    vec4 best = quadrant(at, ivec2(-1, -1))
    best = calmer(best, quadrant(at, ivec2(1, -1)))
    best = calmer(best, quadrant(at, ivec2(-1, 1)))
    best = calmer(best, quadrant(at, ivec2(1, 1)))
    return vec4(linear_srgb_to_srgb(best.rgb), 1.0)"""

# Hatching, given finished. Four layers of stripes, each darker than the last.
HATCH = f"""\
%%gmacs "chelsea" -> "Hatching"
import linear_srgb_color_space
import oklab_color_space

uniform float spacing: hint_range(3, 16) = {SPACING}
uniform float angle: hint_range(0, 180) = {ANGLE}
uniform float thick: hint_range(0.1, 0.8) = {THICK}
uniform float lightest: hint_range(0.5, 1.2) = {LIGHTEST}
uniform int under: hint_enum("White paper", "Photo") = 0

const int WHITE_PAPER = 0
const int PHOTO = 1
const ivec2 SIZE = ivec2({WIDTH}, {HEIGHT})
const float TURN[4] = float[4](0.0, 90.0, -45.0, 45.0)
const float BAND = {BAND}


def light(ivec2 p) -> float:
    ivec2 q = clamp(p, ivec2(0), SIZE - 1)
    return linear_srgb_to_oklab(srgb_to_linear_srgb(src[q].rgb)).x


# The lightness around a pixel, a Gaussian of 2 pixels, so a stroke's width does not flicker with the grain.
def tone(ivec2 at) -> float:
    float total = 0.0
    float weight = 0.0
    for dy in range_inclusive(-6, 6):
        for dx in range_inclusive(-6, 6):
            float w = exp(-float(dx * dx + dy * dy) / 8.0)
            total += w * light(at + ivec2(dx, dy))
            weight += w
    return total / weight


# How much of the pixel at p one family of stripes covers. The stripes run along the
# direction degrees, spacing apart, each width pixels wide with a soft edge of one pixel.
def stripes(vec2 p, float degrees, float width) -> float:
    float a = radians(degrees)
    float across = dot(p, vec2(-sin(a), cos(a))) / spacing
    float distance = abs(fract(across) - 0.5) * spacing
    return clamp(0.5 * width + 0.5 - distance, 0.0, 1.0) * min(width, 1.0)


def pixel(ivec2 at) -> vec4:
    vec2 p = vec2(at) + 0.5
    float t = tone(at)
    float open = 1.0
    for i in range(4):
        float s = clamp((lightest - BAND * float(i) - t) / BAND, 0.0, 1.0)
        open *= 1.0 - stripes(p, angle + TURN[i], thick * spacing * s)
    match under:
        case WHITE_PAPER:
            return vec4(vec3(open), 1.0)
        case PHOTO:
            return vec4(mix(vec3(1.0), src[at].rgb, 0.5) * open, 1.0)"""

SCREEN_DOT_STUB = """\
    # TODO: a disc whose area is coverage of the cell, soft at the edge by one pixel
    return 0.0"""

SOLUTION_SCREEN_DOT = """\
    float r = sqrt(coverage / 3.14159265) * cell
    return clamp(r - length(offset) * cell + 0.5, 0.0, 1.0) * min(2.0 * r, 1.0)"""

HALFTONE = f"""\
%%gmacs "chelsea" -> "Halftone"
import linear_srgb_color_space

uniform float cell: hint_range(4, 24) = {CELL}
uniform int inks: hint_enum("CMYK", "Black only") = 0

const int CMYK = 0
const int BLACK_ONLY = 1
const int LUMA = 4
const ivec2 SIZE = ivec2({WIDTH}, {HEIGHT})
const float SCREEN[4] = float[4](15.0, 75.0, 0.0, 45.0)


# p turned by degrees about the origin.
def rotate(vec2 p, float degrees) -> vec2:
    float a = radians(degrees)
    return vec2(cos(a) * p.x + sin(a) * p.y, -sin(a) * p.x + cos(a) * p.y)


# The linear colour around a point, a Gaussian of a quarter cell, so the whole dot gets one tone.
def around(vec2 centre) -> vec3:
    float sigma = 0.25 * cell
    int r = int(ceil(3.0 * sigma))
    ivec2 middle = ivec2(floor(centre))
    vec3 total = vec3(0.0)
    float weight = 0.0
    for dy in range_inclusive(-r, r):
        for dx in range_inclusive(-r, r):
            float w = exp(-float(dx * dx + dy * dy) / (2.0 * sigma * sigma))
            total += w * srgb_to_linear_srgb(src[clamp(middle + ivec2(dx, dy), ivec2(0), SIZE - 1)].rgb)
            weight += w
    return total / weight


# How much of the paper an ink has to cover. Channels 0 to 2 are cyan, magenta and yellow with
# the black taken out, 3 is the black, LUMA is the one ink of a black and white print.
def coverage_of(vec3 lin, int channel) -> float:
    float m = max(lin.r, max(lin.g, lin.b))
    if channel == LUMA:
        return 1.0 - dot(lin, vec3(0.2126, 0.7152, 0.0722))
    if channel == 3:
        return 1.0 - m
    if m <= 0.0:
        return 0.0
    return 1.0 - lin[channel] / m


# One dot of a screen: offset is the pixel's place in its cell from the cell's centre,
# in cells, each coordinate from -0.5 to 0.5. Returns 1 inside the dot and 0 outside.
def screen_dot(vec2 offset, float coverage) -> float:
{SCREEN_DOT_STUB}


# The ink of one screen at pixel position p: the screen turned by degrees, a dot in each cell.
def ink(vec2 p, float degrees, int channel) -> float:
    vec2 q = rotate(p, degrees) / cell
    vec2 centre = rotate((floor(q) + 0.5) * cell, -degrees)
    return screen_dot(fract(q) - 0.5, clamp(coverage_of(around(centre), channel), 0.0, 1.0))


def pixel(ivec2 at) -> vec4:
    vec2 p = vec2(at) + 0.5
    match inks:
        case CMYK:
            vec3 dots = vec3(ink(p, SCREEN[0], 0), ink(p, SCREEN[1], 1), ink(p, SCREEN[2], 2))
            vec3 paper = (1.0 - dots) * (1.0 - ink(p, SCREEN[3], 3))
            return vec4(linear_srgb_to_srgb(paper), 1.0)
        case BLACK_ONLY:
            return vec4(vec3(linear_srgb_to_srgb(vec3(1.0 - ink(p, 45.0, LUMA)))), 1.0)"""

# The flow: which way the strokes of the picture run, from its structure tensor.
FLOW = f"""\
%%gmacs "chelsea" -> "Flow"
import linear_srgb_color_space
import oklab_color_space

uniform float sigma: hint_range(1, 6) = {FLOW_SIGMA}

const ivec2 SIZE = ivec2({WIDTH}, {HEIGHT})


def light(ivec2 p) -> float:
    ivec2 q = clamp(p, ivec2(0), SIZE - 1)
    return linear_srgb_to_oklab(srgb_to_linear_srgb(src[q].rgb)).x


# Sobel's gradient of the lightness at p.
def gradient(ivec2 p) -> vec2:
    float gx = light(p + ivec2(1, -1)) + 2.0 * light(p + ivec2(1, 0)) + light(p + ivec2(1, 1))
    gx -= light(p + ivec2(-1, -1)) + 2.0 * light(p + ivec2(-1, 0)) + light(p + ivec2(-1, 1))
    float gy = light(p + ivec2(-1, 1)) + 2.0 * light(p + ivec2(0, 1)) + light(p + ivec2(1, 1))
    gy -= light(p + ivec2(-1, -1)) + 2.0 * light(p + ivec2(0, -1)) + light(p + ivec2(1, -1))
    return vec2(gx, gy) / 8.0


# Into red and green the direction of the strokes, a vector of length 1 moved from -1..1 to 0..1.
# Into blue how clear the direction is, 0 where the picture is the same all ways, 1 on a straight edge.
def pixel(ivec2 at) -> vec4:
    int r = int(ceil(3.0 * sigma))
    vec3 j = vec3(0.0)
    float weight = 0.0
    for dy in range_inclusive(-r, r):
        for dx in range_inclusive(-r, r):
            float w = exp(-float(dx * dx + dy * dy) / (2.0 * sigma * sigma))
            vec2 g = gradient(at + ivec2(dx, dy))
            j += w * vec3(g.x * g.x, g.y * g.y, g.x * g.y)
            weight += w
    j /= weight
    float strength = j.x + j.y
    if strength < 0.000001:
        return vec4(1.0, 0.5, 0.0, 1.0)
    float angle = 0.5 * atan(2.0 * j.z, j.x - j.y)
    vec2 tangent = vec2(-sin(angle), cos(angle))
    float clear = sqrt((j.x - j.y) * (j.x - j.y) + 4.0 * j.z * j.z) / strength
    return vec4(0.5 + 0.5 * tangent, clear, 1.0)"""

ACROSS = f"""\
%%gmacs "chelsea" -> "Across" flow=Flow sigma={SIGMA} k={K}
kernel across

import linear_srgb_color_space
import oklab_color_space

image src: readonly
image flow: readonly
image dst

params:
    float sigma: hint_range(0.3, 4)
    float k: hint_range(1.1, 3)

const ivec2 SIZE = ivec2({WIDTH}, {HEIGHT})


def light(vec2 q) -> float:
    ivec2 p = clamp(ivec2(floor(q)), ivec2(0), SIZE - 1)
    return linear_srgb_to_oklab(srgb_to_linear_srgb(src[p].rgb)).x


# Both blurs of the lightness, taken only across the stroke, along the line through the pixel
# that stands at right angles to the flow. Red holds the narrow one, green the wide one.
def across():
    vec2 t = flow[pixel].xy * 2.0 - 1.0
    vec2 n = vec2(-t.y, t.x)
    float wide = k * sigma
    int r1 = int(ceil(3.0 * sigma))
    int r2 = int(ceil(3.0 * wide))
    vec2 total = vec2(0.0)
    vec2 weight = vec2(0.0)
    for i in range_inclusive(-r2, r2):
        vec2 w = exp(-float(i * i) / (2.0 * vec2(sigma * sigma, wide * wide)))
        if abs(i) > r1:
            w.x = 0.0
        total += w * light(vec2(pixel) + 0.5 + float(i) * n)
        weight += w
    vec2 g = total / weight
    dst[pixel] = vec4(g.x, g.y, 0.0, 1.0)"""

FDOG = f"""\
%%gmacs "Across" -> "FDoG" flow=Flow reach={REACH} p={P} epsilon={EPSILON} phi={PHI}
kernel fdog

image src: readonly
image flow: readonly
image dst

params:
    float reach: hint_range(0, 12)
    float p: hint_range(0, 100)
    float epsilon: hint_range(0, 1)
    float phi: hint_range(0, 100)

const ivec2 SIZE = ivec2({WIDTH}, {HEIGHT})


def tangent(vec2 q) -> vec2:
    ivec2 at = clamp(ivec2(floor(q)), ivec2(0), SIZE - 1)
    return flow[at].xy * 2.0 - 1.0


# The two blurs of Across, averaged along the stroke: from the pixel one step at a time along the
# flow, both ways, each step weighted by a Gaussian of reach steps. Then the XDoG of lesson 2.
def fdog():
    vec2 total = src[pixel].rg
    float weight = 1.0
    int steps = int(ceil(2.0 * reach))
    for way in range(2):
        vec2 q = vec2(pixel) + 0.5
        vec2 before = tangent(q) * (1.0 - 2.0 * float(way))
        for i in range_inclusive(1, steps):
            vec2 t = tangent(q)
            if dot(t, before) < 0.0:
                t = -t
            q += t
            before = t
            float w = exp(-float(i * i) / (2.0 * reach * reach + 0.0001))
            total += w * src[clamp(ivec2(floor(q)), ivec2(0), SIZE - 1)].rg
            weight += w
    vec2 g = total / weight
    float s = (1.0 + p) * g.x - p * g.y
    float v = 1.0
    if s < epsilon:
        v = 1.0 + tanh(phi * (s - epsilon))
    dst[pixel] = vec4(vec3(v), 1.0)"""

SKETCH = f"""\
%%gmacs "chelsea" -> "Sketch" halo=32
import linear_srgb_color_space
import oklab_color_space

uniform float sigma: hint_range(1, 8) = {SKETCH_SIGMA}
uniform float gain: hint_range(1, 8) = {SKETCH_GAIN}

const ivec2 SIZE = ivec2({WIDTH}, {HEIGHT})


def light(ivec2 p) -> float:
    ivec2 q = clamp(p, ivec2(0), SIZE - 1)
    return linear_srgb_to_oklab(srgb_to_linear_srgb(src[q].rgb)).x


# The lightness divided by its own blur: 1 on flat ground, below 1 on the dark side of an edge.
def pixel(ivec2 at) -> vec4:
    int r = int(ceil(3.0 * sigma))
    float total = 0.0
    float weight = 0.0
    for dy in range_inclusive(-r, r):
        for dx in range_inclusive(-r, r):
            float w = exp(-float(dx * dx + dy * dy) / (2.0 * sigma * sigma))
            total += w * light(at + ivec2(dx, dy))
            weight += w
    float ratio = light(at) / max(total / weight, 0.0001)
    return vec4(vec3(clamp(1.0 - gain * (1.0 - ratio), 0.0, 1.0)), 1.0)"""


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
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660],
], dtype=np.float32)


def oklab(lin):
    """OKLab of (..., 3) linear RGB."""
    return np.cbrt(lin @ M1.T) @ M2.T


def oklab_to_linear(lab):
    """Linear RGB of (..., 3) OKLab."""
    return (lab @ np.linalg.inv(M2).T) ** 3 @ np.linalg.inv(M1).T


def lightness(img):
    """OKLab L of an (h, w, 4) picture in sRGB, an (h, w) array."""
    return oklab(srgb_to_linear(img[..., :3]))[..., 0]


def rgba(a):
    """An opaque picture from an (h, w) grey or an (h, w, 3) colour array."""
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 2:
        a = np.dstack([a, a, a])
    return np.dstack([a, np.ones(a.shape[:2], np.float32)])


def gaussian_1d(sigma):
    """The weights of a Gaussian of sigma pixels out to 3 sigma, summing to one."""
    r = int(np.ceil(3 * sigma))
    x = np.arange(-r, r + 1)
    k = np.exp(-x ** 2 / (2 * sigma ** 2))
    return k / k.sum()


def blur(a, sigma):
    """a, (h, w, c), blurred by a Gaussian of sigma pixels, rows then columns, edges clamped (lesson 3)."""
    k = gaussian_1d(sigma)
    r = len(k) // 2
    h, w = a.shape[:2]
    p = np.pad(a, ((0, 0), (r, r), (0, 0)), mode="edge")
    a = sum(k[i] * p[:, i:i + w] for i in range(len(k)))
    p = np.pad(a, ((r, r), (0, 0), (0, 0)), mode="edge")
    return sum(k[i] * p[i:i + h] for i in range(len(k)))'''



# --- NumPy ------------------------------------------------------------------------
STYLES_NP = '''\
def xdog(L, sigma, p, epsilon, phi, k=K):
    """XDoG ink drawing of the lightness L, (h, w), from 0 (ink) to 1 (paper)."""
    g1 = blur(L[..., None], sigma)[..., 0]
    g2 = blur(L[..., None], k * sigma)[..., 0]
    s = (1 + p) * g1 - p * g2
    return np.where(s >= epsilon, 1.0, 1.0 + np.tanh(phi * (s - epsilon)))


def toon(img, ink, levels):
    """img's colours with the lightness rounded to levels steps and darkened by ink, (h, w, 3) in sRGB."""
    lab = oklab(srgb_to_linear(img[..., :3]))
    n = levels - 1
    lab[..., 0] = np.floor(np.clip(lab[..., 0], 0, 1) * n + 0.5) / n
    lab *= ink[..., None]
    return linear_to_srgb(np.clip(oklab_to_linear(lab), 0, 1))


def pixel_grid(a):
    """The centres of a's pixels as two (h, w) arrays, x and y."""
    h, w = a.shape[:2]
    return np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)


def stripes(x, y, spacing, degrees, width):
    """How much of each pixel one family of stripes covers, width pixels wide, spacing apart."""
    a = np.radians(degrees)
    across = (-np.sin(a) * x + np.cos(a) * y) / spacing
    distance = np.abs(across % 1.0 - 0.5) * spacing
    return np.clip(0.5 * width + 0.5 - distance, 0, 1) * np.minimum(width, 1.0)


def hatch(L, spacing, angle, thick=THICK, lightest=LIGHTEST):
    """Hatching of the lightness L, (h, w), from 0 (ink) to 1 (paper): four layers of stripes."""
    tone = blur(L[..., None], 2.0)[..., 0]
    x, y = pixel_grid(L)
    paper = np.ones_like(L)
    for i, turn in enumerate(TURN):
        s = np.clip((lightest - BAND * i - tone) / BAND, 0, 1)
        paper *= 1 - stripes(x, y, spacing, angle + turn, thick * spacing * s)
    return paper


def coverage_of(lin, channel):
    """How much of the paper an ink has to cover, (h, w), from the linear colours lin, (h, w, 3)."""
    m = lin.max(axis=-1)
    if channel == LUMA:
        return 1 - lin @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    if channel == 3:
        return 1 - m
    return np.where(m > 0, 1 - lin[..., channel] / np.maximum(m, 1e-6), 0)


def screen_dot(ox, oy, coverage, cell):
    """One dot of a screen: 1 inside the disc whose area is coverage of the cell, soft by a pixel."""
    r = np.sqrt(coverage / np.pi) * cell
    return np.clip(r - np.hypot(ox, oy) * cell + 0.5, 0, 1) * np.minimum(2 * r, 1)


def ink(soft, degrees, channel, cell):
    """The ink of one screen turned by degrees, (h, w): a dot in each cell, its size from the colour at the cell's centre."""
    h, w = soft.shape[:2]
    x, y = pixel_grid(soft)
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    qx, qy = (c * x + s * y) / cell, (-s * x + c * y) / cell
    fx, fy = np.floor(qx), np.floor(qy)
    cx, cy = (fx + 0.5) * cell, (fy + 0.5) * cell
    ix = np.clip(np.floor(c * cx - s * cy), 0, w - 1).astype(int)
    iy = np.clip(np.floor(s * cx + c * cy), 0, h - 1).astype(int)
    return screen_dot(qx - fx - 0.5, qy - fy - 0.5, np.clip(coverage_of(soft[iy, ix], channel), 0, 1), cell)


def halftone(img, cell, inks):
    """img printed with dot screens, (h, w, 3) in sRGB: cyan, magenta, yellow and black, or one black ink."""
    soft = blur(srgb_to_linear(img[..., :3]), 0.25 * cell)
    if inks == "Black only":
        return linear_to_srgb(np.repeat((1 - ink(soft, 45, LUMA, cell))[..., None], 3, axis=-1))
    dots = np.dstack([ink(soft, SCREEN[channel], channel, cell) for channel in range(3)])
    return linear_to_srgb((1 - dots) * (1 - ink(soft, SCREEN[3], 3, cell))[..., None])'''

CONSTANTS_NP = f'''\
K = {K}                              # the second blur is K times wider
THICK, LIGHTEST = {THICK}, {LIGHTEST}    # the hatching's line width and where its first layer starts
BAND = {BAND}                          # the lightness a hatching layer needs to go from nothing to full width
TURN = (0.0, 90.0, -45.0, 45.0)      # each hatching layer's angle from the first
SCREEN = (15.0, 75.0, 0.0, 45.0)     # the angles of the cyan, magenta, yellow and black screens
LUMA = 4                             # the one ink of a black and white print'''

PARAMS_NP = f'''\
PARAMS = {{
    "style": ["Toon", "Ink", "Hatching", "Halftone"],
    "sigma": ({SIGMA}, 0.3, 4.0),
    "p": ({P}, 0.0, 100.0),
    "epsilon": ({EPSILON}, 0.0, 1.0),
    "phi": ({PHI}, 0.0, 100.0),
    "levels": ({LEVELS}, 2, 12),
    "spacing": ({SPACING}, 3.0, 16.0),
    "angle": ({ANGLE}, 0.0, 180.0),
    "on_photo": False,
    "cell": ({CELL}, 4.0, 24.0),
    "inks": ["CMYK", "Black only"],
}}'''

APPLY_NP = '''\
def apply(img, style, sigma, p, epsilon, phi, levels, spacing, angle, on_photo, cell, inks):
    out = img.copy()
    if style == "Toon":
        out[..., :3] = toon(img, xdog(lightness(img), sigma, p, epsilon, phi), levels)
    elif style == "Ink":
        out[..., :3] = xdog(lightness(img), sigma, p, epsilon, phi)[..., None]
    elif style == "Hatching":
        paper = 0.5 + 0.5 * img[..., :3] if on_photo else 1.0
        out[..., :3] = paper * hatch(lightness(img), spacing, angle)[..., None]
    else:
        out[..., :3] = halftone(img, cell, inks)
    return out'''

# The goal, one kernel with all four styles. It is not the code the lesson teaches: the blurs and the
# halftone's colour average sample a few points instead of every pixel, so a slider move stays fast.
GOAL = f"""\
%%gmacs "chelsea" -> "Goal"
import linear_srgb_color_space
import oklab_color_space

uniform int style: hint_enum("Toon", "Ink", "Hatching", "Halftone") = 0
uniform float line_detail: hint_range(0.5, 3) = {SIGMA}
uniform float line_threshold: hint_range(0, 1) = {EPSILON}
uniform float line_softness: hint_range(0, 100) = {PHI}
uniform int toon_levels: hint_range(2, 12) = {LEVELS}
uniform float hatching_spacing: hint_range(3, 16) = {SPACING}
uniform float hatching_angle: hint_range(0, 180) = {ANGLE}
uniform bool hatching_on_photo = false
uniform float halftone_cell: hint_range(4, 24) = {CELL}
uniform bool halftone_black_only = false

const int TOON = 0
const int INK = 1
const int HATCHING = 2
const int HALFTONE = 3
const int LUMA = 4
const ivec2 SIZE = ivec2({WIDTH}, {HEIGHT})
const float K = {K}
const float P = {P}
const float THICK = {THICK}
const float LIGHTEST = {LIGHTEST}
const float BAND = {BAND}
const float TURN[4] = float[4](0.0, 90.0, -45.0, 45.0)
const float SCREEN[4] = float[4](15.0, 75.0, 0.0, 45.0)


def light(ivec2 p) -> float:
    ivec2 q = clamp(p, ivec2(0), SIZE - 1)
    return linear_srgb_to_oklab(srgb_to_linear_srgb(src[q].rgb)).x


# The XDoG drawing, 0 for ink and 1 for paper. Two Gaussian blurs of the lightness at once, k times
# apart. A wide blur reads every second or third pixel only, so a pixel costs about 300 reads.
def lines(ivec2 at) -> float:
    float wide = K * line_detail
    int reach = int(ceil(3.0 * wide))
    int step = max(1, (reach + 8) / 9)
    int n = reach / step
    float near = 3.0 * line_detail
    vec2 total = vec2(0.0)
    vec2 weight = vec2(0.0)
    for j in range_inclusive(-n, n):
        for i in range_inclusive(-n, n):
            vec2 d = vec2(float(i * step), float(j * step))
            float d2 = dot(d, d)
            vec2 w = exp(-d2 / (2.0 * vec2(line_detail * line_detail, wide * wide)))
            if abs(d.x) > near or abs(d.y) > near:
                w.x = 0.0
            total += w * light(at + ivec2(i * step, j * step))
            weight += w
    vec2 g = total / weight
    float s = (1.0 + P) * g.x - P * g.y
    if s >= line_threshold:
        return 1.0
    return 1.0 + tanh(line_softness * (s - line_threshold))


# The photo with its lightness rounded to toon_levels steps and darkened by the lines.
def toon(ivec2 at) -> vec3:
    vec3 lab = linear_srgb_to_oklab(srgb_to_linear_srgb(src[at].rgb))
    float n = float(toon_levels - 1)
    lab.x = floor(clamp(lab.x, 0.0, 1.0) * n + 0.5) / n
    lab *= lines(at)
    vec3 rgb = clamp(oklab_to_linear_srgb(lab), vec3(0.0), vec3(1.0))
    return linear_srgb_to_srgb(rgb)


# The lightness around a pixel, a Gaussian of 2 pixels.
def tone(ivec2 at) -> float:
    float total = 0.0
    float weight = 0.0
    for dy in range_inclusive(-6, 6):
        for dx in range_inclusive(-6, 6):
            float w = exp(-float(dx * dx + dy * dy) / 8.0)
            total += w * light(at + ivec2(dx, dy))
            weight += w
    return total / weight


# How much of the pixel at p one family of stripes covers.
def stripes(vec2 p, float degrees, float width) -> float:
    float a = radians(degrees)
    float across = dot(p, vec2(-sin(a), cos(a))) / hatching_spacing
    float distance = abs(fract(across) - 0.5) * hatching_spacing
    return clamp(0.5 * width + 0.5 - distance, 0.0, 1.0) * min(width, 1.0)


# Four layers of stripes, each turned and each starting at a darker lightness.
def hatching(ivec2 at) -> vec3:
    vec2 p = vec2(at) + 0.5
    float t = tone(at)
    float open = 1.0
    for i in range(4):
        float s = clamp((LIGHTEST - BAND * float(i) - t) / BAND, 0.0, 1.0)
        open *= 1.0 - stripes(p, hatching_angle + TURN[i], THICK * hatching_spacing * s)
    if hatching_on_photo:
        return mix(vec3(1.0), src[at].rgb, 0.5) * open
    return vec3(open)


def rotate(vec2 p, float degrees) -> vec2:
    float a = radians(degrees)
    return vec2(cos(a) * p.x + sin(a) * p.y, -sin(a) * p.x + cos(a) * p.y)


# The linear colour around a point, a Gaussian of a quarter cell sampled on a grid of 5 by 5 points.
def around(vec2 centre) -> vec3:
    float sigma = 0.25 * halftone_cell
    vec3 total = vec3(0.0)
    float weight = 0.0
    for j in range_inclusive(-2, 2):
        for i in range_inclusive(-2, 2):
            float w = exp(-float(i * i + j * j) / 2.0)
            ivec2 q = clamp(ivec2(floor(centre + sigma * vec2(float(i), float(j)))), ivec2(0), SIZE - 1)
            total += w * srgb_to_linear_srgb(src[q].rgb)
            weight += w
    return total / weight


# How much of the paper an ink has to cover. Channels 0 to 2 are cyan, magenta and yellow with the
# black taken out, 3 is the black, LUMA is the one ink of a black and white print.
def coverage_of(vec3 lin, int channel) -> float:
    float m = max(lin.r, max(lin.g, lin.b))
    if channel == LUMA:
        return 1.0 - dot(lin, vec3(0.2126, 0.7152, 0.0722))
    if channel == 3:
        return 1.0 - m
    if m <= 0.0:
        return 0.0
    return 1.0 - lin[channel] / m


# One dot of a screen, 1 inside the disc whose area is coverage of the cell, soft by a pixel.
def screen_dot(vec2 offset, float coverage) -> float:
    float r = sqrt(coverage / 3.14159265) * halftone_cell
    return clamp(r - length(offset) * halftone_cell + 0.5, 0.0, 1.0) * min(2.0 * r, 1.0)


def ink(vec2 p, float degrees, int channel) -> float:
    vec2 q = rotate(p, degrees) / halftone_cell
    vec2 centre = rotate((floor(q) + 0.5) * halftone_cell, -degrees)
    return screen_dot(fract(q) - 0.5, clamp(coverage_of(around(centre), channel), 0.0, 1.0))


def halftone(ivec2 at) -> vec3:
    vec2 p = vec2(at) + 0.5
    if halftone_black_only:
        return vec3(linear_srgb_to_srgb(vec3(1.0 - ink(p, 45.0, LUMA))))
    vec3 dots = vec3(ink(p, SCREEN[0], 0), ink(p, SCREEN[1], 1), ink(p, SCREEN[2], 2))
    return linear_srgb_to_srgb((1.0 - dots) * (1.0 - ink(p, SCREEN[3], 3)))


def pixel(ivec2 at) -> vec4:
    float a = src[at].a
    match style:
        case TOON:
            return vec4(toon(at), a)
        case INK:
            return vec4(vec3(lines(at)), a)
        case HATCHING:
            return vec4(hatching(at), a)
        case HALFTONE:
            return vec4(halftone(at), a)"""


def answer(snippet, lang="python"):
    """A solution block's code fence."""
    return f"```{lang}\n{snippet}\n```"



# --- Title --------------------------------------------------------------------
md(r"""
# Lekce 5: Stylizace, kresba tuší a kreslený vzhled

Na konci lekce budete mít v Kritě filtr se čtyřmi styly: kresbu tuší, kreslený obrázek s plochými barvami a obrysy, šrafování jako v rytině a barevný tiskový rastr jako v novinách. Postavíte ho z věcí, které už umíte: dvou rozmazání z lekce 3, jejich rozdílu, posterizace z lekce 1 a tečkování z lekce 1, jen s tečkami, které mají velikost. Navíc napíšete Kuwaharův filtr, který z fotky udělá malbu, a v nepovinných sekcích si vyzkoušíte čáry vedené podél hran.

Spusťte buňku. Otevře v Sáře nový dokument v sRGB s plátnem 902 × 600 a v něm testovací fotku jako vrstvu `chelsea`.

<details><summary>📖 Proč „kresba tuší“ a ne „čárová kresba“</summary>

Anglické *line art* se doslova překládá jako čárová kresba, což česky nezní. Kernel z druhé sekce dává černé čáry na bílém papíře, tedy to, čemu se česky říká **kresba tuší** (podle materiálu) nebo **perokresba** (podle nástroje). Používáme první, protože XDoG umí i široké tahy štětce.
</details>
""")

code(f"""
import sys
sys.path.insert(0, "lib")            # sara.py and its helpers live in lib/
import numpy as np
import sara

doc, layer, pixels = sara.init(width={WIDTH}, height={HEIGHT})  # a canvas the size of the photo
photo = doc.import_image("imgs/5/chelsea.png")         # a layer called "chelsea"
""")

md(r"""
<details><summary>📖 Nové funkce v této lekci</summary>

**Kernel v buňce `%%gmacs`**

| Zápis | Co dělá | Sekce |
|---|---|---|
| `tanh(x)` | hyperbolický tangens, plynulý schod od −1 do 1 | 2 |
| `image lines: readonly` a `lines=XDoG` na prvním řádku | druhý vstup celého kernelu, vrstva `XDoG` pod jménem `lines` | 3 |
| `int levels` pod `params:` a `levels=5` na prvním řádku | celočíselný parametr celého kernelu a jeho hodnota | 3 |
| `lin[channel]` | složka vektoru podle čísla, které se spočítá za běhu | 6 |
| `atan(y, x)` | úhel vektoru $(x, y)$ v radiánech, od −π do π | 7 |

**NumPy**

| Volání | Co dělá | Sekce |
|---|---|---|
| `np.tanh(a)` | hyperbolický tangens po prvcích | 9 |
| `np.meshgrid(x, y)` | dvě pole `(h, w)` se souřadnicemi $x$ a $y$ každého pixelu | 9 |
| `a % 1.0` | zlomková část po prvcích, jako `fract` v kernelu | 9 |
| `np.hypot(x, y)` | délka vektoru po prvcích | 9 |
| `soft[iy, ix]` | z pole vybere hodnotu na každé dvojici souřadnic najednou | 9 |

</details>
""")

md(r"""
## Cíl lekce

Cíl je jeden kernel se čtyřmi styly. Předpona parametru říká, kterého stylu se týká: `line_` platí pro `Toon` a `Ink`, `toon_` jen pro `Toon`, `hatching_` pro `Hatching` a `halftone_` pro `Halftone`. Aby se posuvníky hýbaly plynule, počítá cíl rozmazání a průměrování barvy z méně bodů, než budete číst vy, takže vypadá skoro stejně jako filtr z lekce, ale kód je jiný.
""")

code(GOAL, hidden=True)

# --- Brainstorming --------------------------------------------------------------
md(r"""
## Jak byste to udělali?

V cíli vyberte styl `Ink`. Fotka se změní na černé čáry na bílém papíře. Jak byste takové čáry z fotky dostali, když kernel zná jen pixel a jeho okolí?

<details><summary>💭 Nápady</summary>

Dobrá kresba tuší splní tři věci: čára je tam, kde se světlost rychle mění, a ne všude, kde je tmavo, čáry jsou souvislé a tenké a zrnitá textura srsti z nich nedělá šum. Každý pixel to musí poznat jen ze svého okolí.

<details><summary>Práh světlosti: tmavé pixely černé, světlé bílé</summary>

Nehodí se. Práh najde tmavé plochy, ne hrany. Oko a nos vyjdou jako černé skvrny, a jakmile práh posunete, srst se rozpadne na zrnitou mapu stínů. Kresba tuší ale kreslí obrysy, ne plochy.

@img(idea_threshold.png, 900, Práh světlosti 0.35 nechá jen oči a nos, práh 0.6 dá zrnitou mapu stínů, v obou případech plochy místo obrysů)

</details>

<details><summary>Sobelův operátor a práh</summary>

Skoro. Sobel z lekce 3 měří, jak rychle se světlost mění, a práh vybere největší změny. Na surové fotce ale vychází i každý chlup, a tak je kresba zrnitá. Po rozmazání jsou čáry klidnější, ale tlusté a bez ladění: velikost detailu, který se ještě počítá jako hrana, určuje jediné číslo, $\sigma$, a tvar čáry nejde měnit. Sekce 1 a 2 to řeší rozdílem dvou rozmazání.

@img(idea_sobel.png, 900, Sobel s prahem dává zrnitou kresbu, po rozmazání čistší ale tlusté dvojité čáry)

</details>

<details><summary>Rozmazat fotku a odečíst ji od originálu</summary>

Ano, a přesně to je sekce 1, jen s dvěma rozmazáními různé šířky místo jednoho. S jedním by zůstaly všechny detaily menší než $\sigma$, i šum.
</details>

<details><summary>Cannyho detektor hran</summary>

Dává nejlepší tenké čáry, ale pracuje po krocích přes celý obrázek: ztenčí hrany a spojí je prahem s hysterezí, který sleduje řetězy pixelů. Pixel by musel znát výsledek sousedů, a to kernel neumí. Pro nepovinnou sekci 7 platí podobná myšlenka, čáry se tam ale vedou podél hran bez sledování řetězů.
</details>

<details><summary>Fotku „vydělit“ její vlastním rozmazáním</summary>

To je trik z Photoshopu zvaný *color dodge*, který z fotky udělá skicu tužkou. Je to nepovinná sekce 8.
</details>

</details>

### A jak by fotka vypadala jako kreslený film?

Přepněte v cíli na `Toon`. Barvy jsou ploché a kolem nich jsou obrysy. Obrysy už máme, a co ty ploché plochy?

<details><summary>💭 Nápady</summary>

<details><summary>Posterizace z lekce 1</summary>

Funguje a je to základ sekce 3. Na fotce s texturou ale každý pixel padne na jednu nebo druhou úroveň, takže srst je zrnitá. Rozmazání před posterizací zrno odstraní, jenže rozmaže i hrany, které mají zůstat ostré.

@img(idea_posterise.png, 900, Posterizace fotky je zrnitá, po rozmazání hladká ale s rozmazanými okraji)

</details>

<details><summary>Vyhladit plochy a nechat hrany</summary>

Přesně to dělá Kuwaharův filtr v sekci 4.
</details>

</details>

### A jak by šla fotka nakreslit šrafami nebo tečkami?

Přepněte v cíli na `Hatching` a `Halftone`. Šrafy jsou čáry, tmavší místa mají hustší, a rastr je mřížka teček, tmavší místa mají větší.

<details><summary>💭 Nápady</summary>

Dobrý nápad musí mít **velikost nebo hustotu značky**, kterou jde zvolit, a ne jen jeden pixel.

<details><summary>Rozptyl z lekce 1 (Bayer)</summary>

Dělá tečky podle světlosti, ale každá je velká jeden pixel a jejich velikost nejde ovlivnit. Vypadá jako zrnění, ne jako tisk.

@img(idea_dither.png, 520, Bayerův rozptyl dává jemné zrnění, tečky jsou jen jeden pixel velké)

</details>

<details><summary>Čáry, kterých je víc, čím je tma</summary>

Ano, to je šrafování. Jedna sada čar ale dá jen dvě úrovně, a tak se vrství víc sad v různých směrech. Sekce 5.
</details>

<details><summary>Mřížka teček s poloměrem podle světlosti</summary>

Ano, rastr. Pro barvu se tiskne čtyřmi barvami zvlášť, každá s vlastní mřížkou. Sekce 6.
</details>

</details>
""")

# --- 1. DoG -----------------------------------------------------------------------
md(r"""
## 1. Rozdíl dvou rozmazání

Potřebujeme místa, kde se světlost rychle mění, a pokud možno jen ty, které odpovídají zvolené velikosti detailu. Sobel a práh to neumí naladit. Rozmazání z lekce 3 je dolní propust: nechá velké tvary a odstraní detaily menší než $\sigma$. Dvě rozmazání s různou $\sigma$ odstraní různě velké detaily, a jejich **rozdíl** (*difference of Gaussians*, DoG) nechá jen detaily mezi oběma velikostmi. Je to **pásmová propust** (*band pass*): ve spektru z lekce 4 ztlumí nízké i vysoké frekvence a nechá pásmo mezi nimi:

$$D = G_\sigma * L - G_{k\sigma} * L, \qquad k \approx 1.6$$

Na hraně je $D$ kladné na světlé straně a záporné na tmavé, uprostřed plochy nula. Podobně vidí hrany i sítnice: buňky reagují na rozdíl středu a okolí, a Marr a Hildreth z toho v roce 1980 udělali detektor hran.

@img(dog_profile.png, 900, Dvě Gaussovy křivky různé šířky, jejich rozdíl ve tvaru mexického klobouku a odezva rozdílu na schodovou hranu se zákmity na obou stranách)

<details><summary>📖 Proč rozdíl dvou rozmazání najde hrany</summary>

Je to skoro **Laplaceův operátor** rozmazaného obrázku, druhá derivace z lekce o konvoluci. Rozmazání s rostoucím $\sigma$ se řídí rovnicí vedení tepla: změna Gaussova jádra s $\sigma$ je jeho Laplace, $\partial G_\sigma / \partial \sigma = \sigma\, \nabla^2 G_\sigma$. Rozdíl pro blízká $\sigma$ a $k\sigma$ je derivace krát krok:

$$G_{k\sigma} - G_\sigma \approx (k\sigma - \sigma)\, \frac{\partial G_\sigma}{\partial \sigma} = (k - 1)\, \sigma^2\, \nabla^2 G_\sigma$$

$$D \approx -(k - 1)\, \sigma^2\, \nabla^2 (G_\sigma * L)$$

DoG je tedy záporný Laplace rozmazané světlosti (*Laplacian of Gaussian*, LoG), až na násobek. Rozmazání napřed potlačí šum, který by druhá derivace jinak zesílila nejvíc. Marr a Hildreth hledali hrany jako místa, kde LoG prochází nulou, a s $k \approx 1.6$ je DoG tvarem LoG nejblíž.
</details>

Kernel níže spočítá obě rozmazání světlosti OKLab L najednou a uloží je do jedné vrstvy: $G_\sigma$ do červené, $G_{k\sigma}$ do zelené a samotnou světlost do modré. Další kernely ji čtou jako tři čísla, ne jako barvu. Na plochách jsou všechna tři čísla skoro stejná, a tak vrstva vypadá šedě, s barevnými lemy u hran.
""")

code(BLURS)

md(r"""
### 🎯 Úkol 1: DoG

Vzorec je výše. Spočítejte rozdíl červené a zelené složky vrstvy `Blurs`, vynásobte ho `gain` a přičtěte k šedé 0.5, aby byly vidět kladné i záporné hodnoty:

$$\text{výsledek} = 0.5 + \text{gain} \cdot (G_\sigma - G_{k\sigma})$$

<details><summary>💡 Nápověda</summary>

`b.r` je $G_\sigma$, `b.g` je $G_{k\sigma}$. Výsledek je šedý, `vec3` ze tří stejných čísel.
</details>

<details><summary>🔑 Řešení</summary>

""" + answer(SOLUTION_DOG) + r"""
</details>
""")

code(DOG, solve=[(DOG_STUB, SOLUTION_DOG)])

md(r"""
<details><summary>❓ Otázka: co dělá <code>sigma</code>?</summary>

Zvětšete `sigma` v buňce `Blurs` na 3 a pak ji zmenšete na 0.5. Jak se změní, které hrany DoG najde? Kernel `DoG` po změně spusťte znovu. Nakonec spusťte buňku `Blurs` znovu, aby měla zase výchozí `sigma` 1.5.

<details><summary>🔑 Odpověď</summary>

S velkou $\sigma$ zůstanou jen obrysy velkých tvarů, hlava a uši, a srst zmizí. S malou $\sigma$ najde DoG i jednotlivé chlupy a šum. $\sigma$ volí velikost detailu, který se počítá jako hrana, a DoG je proto pásmová propust.

</details>

</details>
""")

# --- 2. XDoG ----------------------------------------------------------------------
md(r"""
## 2. XDoG: čáry jako tuší

DoG je šedý obrázek kolem nuly, kresba tuší ale potřebuje černou na bílé. Zkusit prostý práh na $D$ nestačí: dá jen tenké čáry konstantní šířky, bez plynulých stínů. Winnemöller, Kyprianidis a Olsen v roce 2012 proto rozšířili DoG o dvě úpravy (*eXtended DoG*, XDoG). Myšlenky tohoto a následujících sekcí (XDoG, čáry podél hran, šrafování) jsou z videa o Difference of Gaussians, které natočil Acerola (Garrett Gunnell), a z jeho repozitáře [Post-Processing](https://github.com/GarrettGunnell/Post-Processing). Kód je tu napsaný znovu po svém.

Nejdřív se zesílený rozdíl přičte k rozmazanému obrázku, což je zaostření z lekce 3:

$$S = (1 + p)\, G_\sigma * L - p\, G_{k\sigma} * L = G_\sigma * L + p\, D$$

Pak $S$ projde **měkkým prahem**: od $\varepsilon$ výš je papír bílý, pod $\varepsilon$ klesá plynulý schod k černé, strmý podle $\varphi$:

$$T(S) = \begin{cases} 1 & S \ge \varepsilon \\ 1 + \tanh\!\big(\varphi\,(S - \varepsilon)\big) & S < \varepsilon \end{cases}$$

@img(xdog_curve.png, 760, Měkký práh T podle S pro několik strmostí phi, od plynulého přechodu po ostrý schod)

S velkým $\varphi$ je práh ostrý a obrázek černobílý jako dřevoryt. S malým $\varphi$ zůstanou v tmavých místech šedé tóny jako kresba uhlem.

### 🎯 Úkol 2: XDoG

Napište kernel podle vzorců výše.

<details><summary>💡 Nápověda</summary>

1. `float s = (1.0 + p) * b.r - p * b.g`.
2. Výsledek je 1, a jen když je `s` menší než `epsilon`, spočítá se `tanh`.
</details>

<details><summary>🔑 Řešení</summary>

""" + answer(SOLUTION_XDOG) + r"""
</details>
""")

code(XDOG, solve=[(XDOG_STUB, SOLUTION_XDOG)])

md(r"""
Vrstva `XDoG` má vypadat jako levý horní obrázek s výchozími hodnotami, ostatní ukazují, co dělají posuvníky.

@img(xdog_params.png, 900, Šest variant kresby tuší s kočkou s různými hodnotami sigma, p, epsilon a phi)

<details><summary>❓ Otázka: co se stane s <code>p = 0</code> a proč vypadá <code>p = 60</code> s <code>epsilon = 0.9</code> jako negativ?</summary>

<details><summary>🔑 Odpověď</summary>

S $p = 0$ je $S = G_\sigma * L$, rozmazaná světlost, a XDoG je jen práh světlosti: tmavé plochy černé, světlé bílé, žádné obrysy. Velké $p$ zesílí rozdíl, takže na světlé straně hran je $S$ mnohem větší než 1 a na tmavé záporné. S $\varepsilon = 0.9$ je skoro každá plocha pod prahem a zčerná. Bílé zůstanou jen světlé strany hran, které $p$ vytáhlo nad $\varepsilon$: bílé čáry na černém, jako negativ kresby.

</details>

</details>
""")

# --- 3. Toon ----------------------------------------------------------------------
md(r"""
## 3. Kreslený vzhled

Kreslený vzhled (*toon shading*) napodobuje kreslený film: málo odstínů a černé obrysy. Odstíny dá posterizace světlosti z lekce 1, obrysy XDoG. Kernel teď čte **dvě vrstvy**: fotku jako `src` a čáry jako `lines`. Funkce `pixel` umí jen jeden vstup, a tak je kernel napsaný celý, jako sdílený průchod v lekci 3. Řádek `image lines: readonly` přidá druhý obrázek a `lines=XDoG` na prvním řádku buňky říká, která vrstva to je.

@img(toon_steps.png, 900, Fotka kočky, její posterizovaná světlost, čáry XDoG a obojí dohromady jako kreslený obrázek)

Kernel je hotový. Světlost `lab.x` se zaokrouhlí na `levels` úrovní jako v lekci 1 a celé `lab` se vynásobí hodnotou čáry `ink`, takže barva se u čáry ztmaví i odbarví:

$$L' = \frac{\lfloor L\,(n - 1) + 0.5 \rfloor}{n - 1}, \qquad \text{barva} = (L',\, a,\, b) \cdot \text{ink}$$

Spusťte ho a zkuste `levels` v prvním řádku buňky (`levels=5`).
""")

code(TOON)

md(r"""
<details><summary>❓ Otázka: srst kočky je po posterizaci zrnitá. Proč a co by pomohlo?</summary>

<details><summary>🔑 Odpověď</summary>

Světlost srsti se mění pixel od pixelu kolem hranice dvou úrovní, a každý pixel se zaokrouhlí jinam. Pomůže fotku před posterizací vyhladit tak, aby plochy byly hladké a hrany zůstaly ostré. Gaussovo rozmazání by rozmazalo i hrany, a tak přichází Kuwaharův filtr.

</details>

</details>
""")

# --- 4. Kuwahara ------------------------------------------------------------------
md(r"""
## 4. Kuwahara: fotka jako malba

Kuwaharův filtr z roku 1976 vyhladí plochy a hrany nechá ostré. Kolem pixelu vezme čtyři čtverce, každý s pixelem v jednom rohu. Spočítá v každém průměrnou barvu a rozptyl jasu, a pixel dostane průměr čtverce s **nejmenším rozptylem**, toho nejklidnějšího. U hrany je to čtverec, který leží celý na jedné straně, a hrana se proto nerozmaže. Výsledek vypadá jako olejomalba, s plochými tahy štětce.

@img(kuwahara.png, 900, Čtyři čtverce kolem pixelu na hraně, ten s nejmenším rozptylem zvýrazněný, a výřez fotky před filtrem a po něm)

Rozptyl je průměr čtverců minus čtverec průměru:

$$\operatorname{var} = \frac{1}{n}\sum l^2 - \Big(\frac{1}{n}\sum l\Big)^2$$

### 🎯 Úkol 3: Kuwahara

Doplňte funkci `quadrant`. Čtverec má stranu `radius + 1` a vede od `at` ve směru `towards`: pixel `at + towards * ivec2(dx, dy)` pro `dx` a `dy` od 0 do `radius`. Vrací průměrnou barvu v lineárním RGB a ve složce `.a` rozptyl jasu $l = 0.2126\,R + 0.7152\,G + 0.0722\,B$. Funkce `calmer` a `pixel` jsou hotové.

<details><summary>💡 Nápověda</summary>

1. Tři součty: barev `total`, jasů `s1` a čtverců jasů `s2`.
2. Barvu převeďte do lineárního RGB hned při čtení.
3. Počet pixelů je `(radius + 1) * (radius + 1)`, ve `float`.
</details>

<details><summary>🔑 Řešení</summary>

""" + answer(SOLUTION_QUADRANT) + r"""
</details>
""")

code(KUWAHARA, solve=[(QUADRANT_STUB, SOLUTION_QUADRANT)])

md(r"""
<details><summary>📖 Kreslený vzhled z malby a řetěz průchodů</summary>

V buňce kernelu `Toon` změňte první řádek na

```
%%gmacs "Kuwahara" -> "Toon" lines=XDoG levels=5
```

a spusťte ji znovu. Zkuste také `Blurs` z vrstvy `Kuwahara`, aby i čáry byly klidnější: první řádek `%%gmacs "Kuwahara" -> "Blurs" halo=32` a pak znovu `XDoG` a `Toon`. Tak vypadá řetěz průchodů, *pipeline*: každý kernel čte vrstvu, kterou zapsal předchozí.

@img(pipeline.png, 900, Řetěz vrstev lekce: fotka, Kuwahara, rozmazání, XDoG a toon, se šipkami od vrstvy ke kernelu, který ji čte)

</details>

<details><summary>❓ Otázka: proč se s <code>radius = 8</code> objeví hranaté skvrny a proč se Kuwahara v praxi mění?</summary>

<details><summary>🔑 Odpověď</summary>

Čtverce mají rovné strany a pevné směry, takže výsledek se skládá z kousků čtverců, hlavně tam, kde se nejklidnější čtverec mění skokem. Varianty, které se používají, mění tvar oblasti: kruhové výseče s plynulými vahami (*generalised Kuwahara*) a výseče protažené podél hrany (*anisotropic Kuwahara*), jak to ukazuje Acerola v repozitáři Post-Processing. Směr hrany spočítáme v nepovinné sekci 7.

</details>

</details>
""")

# --- 5. Hatching ------------------------------------------------------------------
md(r"""
## 5. Šrafování

Šrafování (*hatching*) kreslí tmavá místa hustšími čarami, jako rytina nebo pero. Potřebujeme čáry, jejichž hustota sleduje světlost fotky. **Naivní pokus:** jedna sada rovnoběžných čar, nakreslená všude, kde je světlost pod 0.5. Vyjdou dvě úrovně, čáry nebo papír, a okraj mezi nimi je ostrý. Lepší je vrstvit sady čar v různých směrech, každou od tmavšího tónu než předchozí, a čáry tam, kde se tón blíží prahu vrstvy, nechat zeslabit.

@img(hatch_layers.png, 1000, Jedna sada čar pod prahem světlosti dává ostré skvrny, čtyři vrstvy čar v různých směrech dají plynulé stínování)

Pro vrstvu $i = 0, 1, 2, 3$ kernel udělá tři věci. Tón $L$ je světlost OKLab rozmazaná Gaussem o 2 pixelech, aby šířka čáry neblikala se zrnem fotky. Vrstva začíná na světlosti $T_i = \text{lightest} - B\,i$, kde $B$ je pásmo tónů jedné vrstvy, a hustota roste od 0 do 1:

$$s_i = \operatorname{clamp}\!\Big(\frac{T_i - L}{B},\, 0,\, 1\Big), \qquad w_i = \text{thick} \cdot \text{spacing} \cdot s_i$$

Čáry vrstvy $i$ jedou ve směru $\text{angle} + (0, 90, -45, 45)_i$ stupňů, vždy po `spacing` pixelech, široké $w_i$. Pokrytí pixelu čarami je vzdálenost k nejbližší ose čáry, převedená na měkký okraj o jednom pixelu:

$$\text{stripe} = \operatorname{clamp}\!\big(\tfrac{w}{2} + 0.5 - d,\, 0,\, 1\big) \cdot \min(w, 1)$$

Poslední činitel zajistí, že čára šířky nula nic nekreslí. Papír propustí světlo vrstvou po vrstvě:

$$\text{papír} = \prod_{i=0}^{3} (1 - \text{stripe}_i)$$

Kernel je hotový. Zkuste posuvníky a přepínač `under`: `Photo` vezme pod čáry polovinu barvy fotky.

<details><summary>📖 Odkud je to</summary>

Vrstvy s postupně tmavším prahem jsou zjednodušení myšlenky *tonal art maps* (Praun a kol., 2001). Šrafování má i Acerola v repozitáři Post-Processing. Tahle verze je napsaná po svém a čáry se počítají vzorcem, takže jdou měnit posuvníkem.
</details>
""")

code(HATCH)

md(r"""
<details><summary>❓ Otázka: proč jsou čtyři směry zrovna 0, 90, −45 a 45 stupňů od prvního?</summary>

<details><summary>🔑 Odpověď</summary>

První dvě vrstvy jsou na sebe kolmé a dají čtvercovou síť, další dvě jsou o 45 stupňů otočené a přidají do ní úhlopříčky, takže nejtmavší místa se zaplní. Úhly jsou věc vkusu, zkuste v kernelu jiné hodnoty `TURN`.

</details>

</details>
""")

# --- 6. Halftone ------------------------------------------------------------------
md(r"""
## 6. Rastr: půltóny jako v novinách

Tiskárna nemá šedé barvy, jen černou na bílém papíře. Tmavší místa tedy dostanou větší tečky na pravidelné mřížce a oko je z dálky smíchá do šedi. Tomu se říká **rastr** (*halftone*, česky také půltónový tisk). Barevný tisk má čtyři barvy, azurovou, purpurovou, žlutou a černou (*CMYK*), každou s vlastním rastrem. **Naivní pokus:** všechny čtyři mřížky stejně, bez natočení. Tečky se pak kreslí přes sebe a drobné posuny mezi nimi dají vzory, které se mění po celém obrázku (*moiré*). Tisk to řeší natočením každé mřížky o jiný úhel, tradičně 15, 75, 0 a 45 stupňů.

@img(halftone_inks.png, 1000, Čtyři barvy rastru zvlášť, pak všechny čtyři mřížky v nule stupňů a nakonec natočené mřížky, které tvoří rozetu bez moiré)

**Kolik barvy.** Z lineárního RGB se barvy spočítají tiskovým způsobem. Černá bere největší složku, ostatní barvy zbytek:

$$m = \max(R, G, B), \qquad K = 1 - m, \qquad C = 1 - \frac{R}{m},\quad M = 1 - \frac{G}{m},\quad Y = 1 - \frac{B}{m}$$

Světlo, které papír po tisku odrazí, je součin toho, co každá barva propustí:

$$\text{barva} = (1 - C,\, 1 - M,\, 1 - Y)\,(1 - K)$$

Výpočet jde v lineárním světle, protože tečky skutečně sčítají světlo, které papír odrazí. Černobílý tisk má jedinou barvu s pokrytím $1 - L_\text{lin}$, kde $L_\text{lin}$ je jas v lineárním světle.

**Kolik papíru pokryje tečka.** Pokrytí $c$ je podíl buňky, který má tečka zakrýt. Plocha kruhu je $\pi r^2$ a plocha buňky $\text{cell}^2$, takže poloměr kruhu je

$$r = \text{cell}\,\sqrt{c / \pi}$$

Vzdálenost $d$ od středu je v pixelech `length(offset) * cell`, a hrana tečky je měkká o jeden pixel:

$$\text{dot} = \operatorname{clamp}(r - d + 0.5,\, 0,\, 1) \cdot \min(2r,\, 1)$$

Poslední činitel řeší tečku s nulovým pokrytím: bez něj by se v jejím středu kreslil poloviční pixel a prázdné barvy by zašpinily obrázek.

Zbytek kernelu je hotový. Pokrytí čte `coverage_of` ze světla kolem středu buňky (`around` rozmaže na čtvrt buňky, aby celá tečka dostala jednu barvu), `ink` otočí souřadnice mřížky a najde buňku a `pixel` složí čtyři barvy. Chybí jedno: tečka.

### 🎯 Úkol 4: tečka rastru

Doplňte `screen_dot(offset, coverage)`. `offset` je poloha pixelu v buňce od jejího středu, každá souřadnice v rozmezí od −0.5 do 0.5, v jednotkách buňky. Vraťte 1 uvnitř tečky a 0 mimo, s měkkým okrajem podle vzorců výše.

<details><summary>💡 Nápověda</summary>

1. `float r = sqrt(coverage / 3.14159265) * cell` je poloměr v pixelech.
2. Vzdálenost je `length(offset) * cell`.
3. `clamp(r - d + 0.5, 0.0, 1.0)` je okraj. Pokrytí 0 ošetřete činitelem `min(2.0 * r, 1.0)`.
</details>

<details><summary>🔑 Řešení</summary>

""" + answer(SOLUTION_SCREEN_DOT) + r"""
</details>
""")

code(HALFTONE, solve=[(SCREEN_DOT_STUB, SOLUTION_SCREEN_DOT)])

md(r"""
Zkuste `cell` 4 a 20 a přepněte `inks` na `Black only`. Je to jedna mřížka pod 45 stupni jako u novin.

<details><summary>❓ Otázka: proč je střed buňky rozmazaný a ne jeden pixel?</summary>

<details><summary>🔑 Odpověď</summary>

Jeden pixel uprostřed buňky by byl šum fotky, a každá tečka by dostala jinou velikost, než odpovídá jejímu okolí. Rozmazání na čtvrt buňky dá středu průměr z celého místa, které tečka zastupuje, a tečky pak sledují tvary, ne zrno.

</details>

</details>
""")

# --- 7. Flow ----------------------------------------------------------------------
md(r"""
## 7. Čáry podél hran (nepovinné)

XDoG z druhé sekce rozmazává do všech směrů, a tak kolem hran zůstávají přerušované kousky. Kdyby se čáry vedly **podél** hran, spojily by se do plynulých tahů štětce. Tomu se říká DoG řízené tokem (*flow-based DoG*, Kang, Lee a Chui, 2007, v Acerolově repozitáři také). Potřebujeme pro každý pixel vědět, kam hrana směřuje: **tečnu hran** (*edge tangent flow*).

Směr hrany je kolmý na gradient, který už umíme ze Sobela z lekce 3. Přímo ho ale průměrovat nejde: gradient míří na jednu stranu hrany a na druhé na opačnou, takže by se zrušily. Průměruje se proto jeho čtverec, **strukturní tenzor** (*structure tensor*), který na znaménku nezávisí.

<details><summary>📖 Strukturní tenzor</summary>

Ze Sobelova gradientu $g = (g_x, g_y)$ se v každém pixelu spočítá matice a Gaussem se zprůměruje v okolí o $\sigma$ pixelech:

$$J = G_\sigma * \begin{pmatrix} g_x^2 & g_x g_y \\ g_x g_y & g_y^2 \end{pmatrix}$$

Směr, ve kterém se světlost mění nejvíc, má úhel

$$\theta = \tfrac{1}{2}\, \operatorname{atan2}\!\big(2 J_{xy},\ J_{xx} - J_{yy}\big)$$

a tečna je k němu kolmá, $t = (-\sin\theta,\ \cos\theta)$. Jak jasný směr je, říká $\sqrt{(J_{xx} - J_{yy})^2 + 4 J_{xy}^2} \,/\, (J_{xx} + J_{yy})$: 1 na rovné hraně a 0 tam, kde se obraz mění stejně do všech stran.
</details>

@img(flow_field.png, 760, Úsečky tečny hran na výřezu oka kočky, oranžové tam, kde je směr jasný, modré tam, kde ne)

Kernel `Flow` zapíše tečnu do červené a zelené složky (z $-1$ až $1$ převedenou na 0 až 1) a jasnost směru do modré.
""")

code(FLOW)

md(r"""
Teď dvě rozmazání, jen **napříč** tahu: kernel `Across` vezme čáru kolmou k toku a rozmaže jen po ní. Pak kernel `FDoG` ta dvě rozmazání **zprůměruje podél** toku, krok po kroku oběma směry, a z výsledku udělá XDoG ze sekce 2. Oba kernely jsou celé a kód mají sbalený.
""")

code(ACROSS, hidden=True)

code(FDOG, hidden=True)

md(r"""
Vrstva `FDoG` má stejné čáry jako `XDoG`, ale tahy jsou souvislé a vedou podél srsti. Zkuste `reach` v prvním řádku buňky `FDoG` (délka tahu) a `sigma` v buňce `Across`.

<details><summary>❓ Otázka: proč se rozmazání dělají napříč a pak podél a ne obě najednou?</summary>

<details><summary>🔑 Odpověď</summary>

Napříč tahu je třeba najít hranu, a tak se obraz rozmazává jen tam. Podél tahu hrana nic neříká, a tak se tam průměruje bez obav, čímž se spojí přerušované kousky. Obyčejné Gaussovo rozmazání do všech směrů by hranu rozmazalo i podél ní.

</details>

</details>
""")

# --- 8. Sketch --------------------------------------------------------------------
md(r"""
## 8. Skica tužkou (nepovinné)

Poslední styl je trik z Photoshopu, **color dodge** s rozmazanou kopií. Světlost se vydělí vlastním rozmazáním: na ploché ploše je podíl 1 (bílá), na tmavé straně hrany pod 1:

$$\text{výsledek} = 1 - g\,\Big(1 - \frac{L}{G_\sigma * L}\Big)$$

Kernel je hotový. Zkuste `sigma`, velikost detailu, který se kreslí, a `gain`, tmavost tahů.
""")

code(SKETCH)

# --- 9. NumPy and Krita -----------------------------------------------------------
md(r"""
## 9. Do Krity

Krita pouští filtry v NumPy, takže čtyři styly jsou níže přepsané do NumPy. Spusťte obě buňky. První obsahuje konstanty, převody barev z úvodní lekce a rozmazání `blur` z lekce 3, druhá styly `xdog`, `toon`, `hatch`, `halftone` a `apply`.
""")

code(CONSTANTS_NP + "\n\n\n" + HELPERS_NP)

code(STYLES_NP + "\n\n\n" + PARAMS_NP + "\n\n\n" + APPLY_NP)

code("""
sara.live(apply, PARAMS, source="chelsea", target="Filter NumPy")
""")

md(r"""
Vrstva `Filter NumPy` vypadá skoro stejně jako `Goal` a jako vrstvy kernelů. Rozdíly jsou jen v zaokrouhlení a v tom, že `Goal` počítá rozmazání z méně bodů.

<details><summary>📖 Co se v NumPy mění oproti kernelu</summary>

Kernel počítá jeden pixel, NumPy celé pole najednou. Tím se liší hlavně tyhle řádky:

| Kernel | NumPy |
|---|---|
| `at`, `vec2(at) + 0.5` | `pixel_grid`: dvě pole `x` a `y` se souřadnicemi středů všech pixelů |
| `fract(x)` | `x % 1.0` |
| cyklus `for i in range(4)` nad vrstvami šrafování | tentýž cyklus v Pythonu, každý průchod ale počítá celý obrázek |
| `around(centre)`: Gaussův průměr kolem středu buňky pro každý pixel zvlášť | `blur` celého obrázku jednou, pak výběr hodnoty ve středu buňky přes `soft[iy, ix]` |
| `lin[channel]` | `lin[..., channel]`, poslední osa jsou kanály |
| `clamp(p, ivec2(0), SIZE - 1)` na okrajích | `np.clip(..., 0, w - 1)` pro souřadnice a `mode="edge"` v `blur` |
| `match inks:` a `match under:` | `if inks == "Black only"` a `on_photo` jako zaškrtávací políčko |
| `tanh(x)`, `length(v)` | `np.tanh(x)`, `np.hypot(x, y)` |

Čáry podél hran a skica v NumPy nejsou: tok potřebuje pro každý pixel sledovat dráhu přes obraz, a to je v NumPy pomalé.

</details>

### 🎯 Úkol 5: filtr v Kritě

V `pga_filter/effect.py` nahraďte všechno pod řádkem `import numpy as np` obsahem obou buněk s kódem výše a nastavte `TITLE = "Stylisation"`. Vyzkoušejte všechny čtyři styly na portrétu a na krajině.

<details><summary>ℹ️ Šrafování nebo rastr v Kritě vypadá jinak než ve zmenšeném náhledu?</summary>

Náhled filtru ve zmenšené velikosti zmenší i obrázek, ale `spacing` a `cell` jsou v pixelech, takže čáry a tečky se v něm zdají hrubší. Přepněte náhled na „Centre at full size“.
</details>
""")

# --- Summary ------------------------------------------------------------------
md(r"""
## Shrnutí

- Rozdíl dvou Gaussových rozmazání je pásmová propust: nechá detaily mezi dvěma velikostmi. XDoG ho zesílí a pošle přes měkký práh, výsledek jsou čáry tuší.
- Kreslený vzhled je posterizovaná světlost krát čáry, Kuwahara vyhladí plochy a nechá hrany.
- Šrafování vrství sady čar v různých směrech, každá se zapíná od tmavšího tónu.
- Rastr kreslí tečku s plochou podle pokrytí, každá barva CMYK na mřížce jiného úhlu, aby nevzniklo moiré.
- Tok hran ze strukturního tenzoru dovolí vést čáry podél hran.

<details><summary>📖 Co jsme vynechali</summary>

- **Cannyho detektor hran** a hystereze: pracují po krocích přes celý obrázek.
- **Znaky ASCII** místo teček. Potřebují obrázek s písmeny jako druhý vstup, a to teď umíte.
- **Paleta.** Místo posterizace světlosti se barvy dají zaokrouhlit na paletu několika barev, vybranou ručně nebo shlukováním (k-means).
- **Anizotropní Kuwahara** s tokem hran místo čtverců.
- **Tok a skica v NumPy**, tedy v Kritě.

</details>

<details><summary>🎯 Bonusové úkoly</summary>

1. **Barevné čáry.** Místo černé kreslete čáry barvou: tmavší verzí barvy plochy pod nimi, nebo pevnou barvou z výběru barvy (`source_color`).
2. **Šrafy podél toku.** Do `Hatching` pošlete vrstvu `Flow` a otáčejte čáry podle tečny hran místo pevného úhlu.
3. **Čárový rastr.** Nahraďte `screen_dot` pruhem, jehož šířka sleduje pokrytí, jako u rytiny.
4. **Rychlá Kuwahara.** Součty ve čtvercích jdou spočítat z integrálního obrazu (*summed-area table*) v konstantním čase, nezávisle na poloměru. Napište to v NumPy přes `np.cumsum`.

</details>

<details><summary>📖 Nápady na semestrální práci</summary>

- Plugin „komiks“: kreslený vzhled s půltónovými stíny, barevnými obrysy a volbou palety.
- Anizotropní Kuwaharův filtr s polem směrů ze strukturního tenzoru, jako malířský filtr.
- Generátor omalovánek: čáry XDoG, uzavřené plochy a jejich čísla podle barvy.

</details>
""")


def build():
    lesson_writer.write(cells, OUT, "l5")


if __name__ == "__main__":
    build()
