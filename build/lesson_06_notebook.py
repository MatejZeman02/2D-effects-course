"""Builds the lesson 6 notebook in the lecture folder above this one.

The notebook is generated: edit this file and run it again, never the .ipynb
by hand, or the next build overwrites the edit. The pictures and the test
picture rocket_hdr.npz come from lesson_06_diagrams.py, which turns
scikit-image's rocket photo into linear light with its lamps made a hundred
times brighter, as imgs/6/SOURCES.md says.

Markdown cells are raw strings, so LaTeX keeps single backslashes and braces.
A picture from imgs/6 is written as @img(file, width, alt text).

The solutions are constants near the top, so the markdown shows exactly the
code that test_lesson_06.py runs.
"""
import re
import textwrap
from pathlib import Path

import lesson_writer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "lesson_06_hdr.ipynb"
IMG = "imgs/6"
WIDTH, HEIGHT = 640, 427      # the photo's size and the canvas's
KEY = 0.12                    # the grey auto exposure aims the mean at
THRESHOLD, SIGMA, STRENGTH = 2.0, 6.0, 0.1
SIGMA_R, ALPHA, BETA = 1.0, 1.0, 0.4   # the local Laplacian filter of section 6, in stops
cells = []


def _pictures(text):
    return re.sub(r"@img\(([^,]+), (\d+), ([^)]*)\)",
                  lambda m: f'<img src="{IMG}/{m[1]}" width="{m[2]}" alt="{m[3]}">', text)


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": _pictures(text).strip("\n").splitlines(keepends=True)})


def code(text, hidden=False, solve=()):
    metadata = {"jupyter": {"source_hidden": True}} if hidden else {}
    cell = {"cell_type": "code", "metadata": metadata, "execution_count": None, "outputs": [],
            "source": lesson_writer.lines(text)}
    if solve:
        cell["solution"] = lesson_writer.solved(text, solve)
    cells.append(cell)


ACES_GLSL = """\
def aces(vec3 x) -> vec3:
    return clamp(x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14), vec3(0.0), vec3(1.0))"""

# --- The kernels ------------------------------------------------------------------
EXPOSURE = """\
%%gmacs "HDR" -> "Exposure"
import linear_srgb_color_space

uniform float stops: hint_range(-8, 8) = 0.0


# The layer holds linear light. Exposure scales it by 2^stops, the screen cuts it at 1.
def pixel(ivec2 at) -> vec4:
    vec3 c = src(at).rgb * exp2(stops)
    return vec4(linear_srgb_to_srgb(clamp(c, vec3(0.0), vec3(1.0))), 1.0)"""

TONE_STUB = """\
    # TODO: Reinhard for curve 1 and the ACES fit for curve 2
    return clamp(x, vec3(0.0), vec3(1.0))"""

SOLUTION_TONE = """\
    if curve == 1:
        return x / (1.0 + x)
    if curve == 2:
        return clamp(x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14), vec3(0.0), vec3(1.0))
    return clamp(x, vec3(0.0), vec3(1.0))"""

TONE = f"""\
%%gmacs "HDR" -> "Tone"
import linear_srgb_color_space

uniform float stops: hint_range(-8, 8) = 0.0
uniform int curve: hint_enum("Clip", "Reinhard", "ACES") = 2


def tone(vec3 x) -> vec3:
    \"\"\"Linear light from 0 up, mapped into 0 to 1 by the curve.\"\"\"
{TONE_STUB}


def pixel(ivec2 at) -> vec4:
    vec3 c = src(at).rgb * exp2(stops)
    return vec4(linear_srgb_to_srgb(tone(c)), 1.0)"""

LOG_STUB = """\
    # TODO: the logarithm of DELTA plus the pixel's luminance, in all three channels
    return vec4(0.0, 0.0, 0.0, 1.0)"""

SOLUTION_LOG = """\
    float y = dot(src(at).rgb, vec3(0.2126, 0.7152, 0.0722))
    return vec4(vec3(log(DELTA + y)), 1.0)"""

LOG_LUMINANCE = f"""\
%%gmacs "HDR" -> "Log luminance"
const float DELTA = 0.0001    # so that black is not log 0


def pixel(ivec2 at) -> vec4:
{LOG_STUB}"""

AUTO = f"""\
%%gmacs "HDR" -> "Auto" exposure={{exposure}}
import linear_srgb_color_space

uniform float exposure: hint_range(0, 16) = 1.0


{ACES_GLSL}


# The exposure comes from the cell above, written into the first line by {{exposure}}.
def pixel(ivec2 at) -> vec4:
    return vec4(linear_srgb_to_srgb(aces(src(at).rgb * exposure)), 1.0)"""

BRIGHT = f"""\
%%gmacs "HDR" -> "Bright"
uniform float threshold: hint_range(0, 16) = {THRESHOLD}


# Only the light above the threshold, its colour kept: what is bright enough to glow.
def pixel(ivec2 at) -> vec4:
    vec3 c = src(at).rgb
    float y = dot(c, vec3(0.2126, 0.7152, 0.0722))
    return vec4(c * max(y - threshold, 0.0) / max(y, 0.0001), 1.0)"""

BRIGHT_X = f"""\
%%gmacs "Bright" -> "Bright X" halo=64
uniform float sigma: hint_range(1, 16) = {SIGMA}


# The convolution lesson's first pass, without the sRGB conversions: the layer is linear light already.
def pixel(ivec2 at) -> vec4:
    int r = int(ceil(3.0 * sigma))
    vec3 total = vec3(0.0)
    float weight = 0.0
    for dx in range_inclusive(-r, r):
        float k = exp(-float(dx * dx) / (2.0 * sigma * sigma))
        total += k * src(at + ivec2(dx, 0)).rgb
        weight += k
    return vec4(total / weight, 1.0)"""

BLOOM = f"""\
%%gmacs "Bright X" -> "Bloom" halo=64
uniform float sigma: hint_range(1, 16) = {SIGMA}


def pixel(ivec2 at) -> vec4:
    int r = int(ceil(3.0 * sigma))
    vec3 total = vec3(0.0)
    float weight = 0.0
    for dy in range_inclusive(-r, r):
        float k = exp(-float(dy * dy) / (2.0 * sigma * sigma))
        total += k * src(at + ivec2(0, dy)).rgb
        weight += k
    return vec4(total / weight, 1.0)"""

COMPOSITE_STUB = """\
    # TODO: the glow added to the light, times strength, before the exposure and the curve
    vec3 mapped = aces(c * exposure)"""

SOLUTION_COMPOSITE = """\
    vec3 mapped = aces((c + strength * glow) * exposure)"""

COMPOSITE = f"""\
%%gmacs "HDR" -> "Glow" bloom=Bloom exposure={{exposure}} strength={STRENGTH}
kernel glow

import linear_srgb_color_space

image src: readonly
image bloom: readonly
image dst

params:
    float exposure: hint_range(0, 16)
    float strength: hint_range(0, 1)


{ACES_GLSL}


@workgroup_size(8, 8)
def glow():
    vec3 c = image_load(src, pixel).rgb
    vec3 glow = image_load(bloom, pixel).rgb
{COMPOSITE_STUB}
    image_store(dst, pixel, vec4(linear_srgb_to_srgb(mapped), 1.0))"""

FIT_STUB = """\
    # TODO: lightness at most 1, then the largest share of the chroma that lands inside the cube
    return clamp(c, vec3(0.0), vec3(1.0))"""

SOLUTION_FIT = """\
    if inside(c):
        return c
    vec3 lab = linear_srgb_to_oklab(c)
    lab.x = min(lab.x, 1.0)
    float lo = 0.0
    float hi = 1.0
    for i in range(0, 16):
        float mid = 0.5 * (lo + hi)
        if inside(oklab_to_linear_srgb(vec3(lab.x, mid * lab.yz))):
            lo = mid
        else:
            hi = mid
    return clamp(oklab_to_linear_srgb(vec3(lab.x, lo * lab.yz)), vec3(0.0), vec3(1.0))"""

GAMUT = f"""\
%%gmacs "HDR" -> "Gamut" exposure={{exposure}}
import linear_srgb_color_space
import oklab_color_space

uniform float exposure: hint_range(0, 16) = 1.0
uniform int mode: hint_enum("Per channel", "Clip", "OKLab") = 2


{ACES_GLSL}


def inside(vec3 c) -> bool:
    return all(greaterThanEqual(c, vec3(0.0))) and all(lessThanEqual(c, vec3(1.0)))


def fit(vec3 c) -> vec3:
    \"\"\"c, which may lie outside the cube from 0 to 1, moved inside with its OKLab hue kept.\"\"\"
{FIT_STUB}


def pixel(ivec2 at) -> vec4:
    vec3 c = src(at).rgb * exposure
    if mode == 0:
        return vec4(linear_srgb_to_srgb(aces(c)), 1.0)
    # the luminance through the curve, the colour scaled with it, so its ratios stay
    float y = dot(c, vec3(0.2126, 0.7152, 0.0722))
    vec3 scaled = c * aces(vec3(y)).x / max(y, 0.0001)
    if mode == 1:
        return vec4(linear_srgb_to_srgb(clamp(scaled, vec3(0.0), vec3(1.0))), 1.0)
    return vec4(linear_srgb_to_srgb(fit(scaled)), 1.0)"""

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


def luminance(lin):
    """The luminance Y of (..., 3) linear RGB."""
    return lin @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


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
    """a, (h, w, c), blurred by a Gaussian of sigma pixels, rows then columns, edges clamped (the convolution lesson)."""
    k = gaussian_1d(sigma)
    r = len(k) // 2
    h, w = a.shape[:2]
    p = np.pad(a, ((0, 0), (r, r), (0, 0)), mode="edge")
    a = sum(k[i] * p[:, i:i + w] for i in range(len(k)))
    p = np.pad(a, ((r, r), (0, 0), (0, 0)), mode="edge")
    return sum(k[i] * p[i:i + h] for i in range(len(k)))'''

CURVES_NP = '''\
def reinhard(x):
    return x / (1 + x)


def aces(x):
    """Narkowicz's fit of the ACES film curve, from 0 to 1."""
    return np.clip(x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14), 0, 1)'''

AUTO_NP_GIVEN = f'''\
def auto_exposure(lin, key={KEY}):
    """The exposure that brings the picture's mean log luminance to key."""
    # TODO: the geometric mean of DELTA + Y over every pixel, then key divided by it
    return 1.0'''

SOLUTION_AUTO_NP = f'''\
def auto_exposure(lin, key={KEY}):
    """The exposure that brings the picture's mean log luminance to key."""
    mean_log = np.log(1e-4 + luminance(lin)).mean()
    return key / np.exp(mean_log)'''

PARAMS_NP = f'''\
PARAMS = {{
    "auto": True,
    "key": ({KEY}, 0.01, 1.0),
    "stops": (0.0, -8.0, 8.0),
    "threshold": ({THRESHOLD}, 0.0, 16.0),
    "sigma": ({SIGMA}, 1.0, 16.0),
    "strength": ({STRENGTH}, 0.0, 1.0),
    "curve": ["ACES", "Reinhard", "Clip"],
    "encode": True,
}}'''

APPLY_GIVEN = '''\
def apply(img, auto, key, stops, threshold, sigma, strength, curve, encode):
    out = img.copy()
    # TODO: exposure, the bloom, the tone curve and, when encode is on, sRGB
    return out'''

SOLUTION_APPLY = '''\
def apply(img, auto, key, stops, threshold, sigma, strength, curve, encode):
    out = img.copy()
    lin = np.maximum(img[..., :3], 0.0)
    exposure = 2.0 ** stops * (auto_exposure(lin, key) if auto else 1.0)
    y = luminance(lin)
    bright = lin * (np.maximum(y - threshold, 0.0) / np.maximum(y, 1e-4))[..., None]
    lin = (lin + strength * blur(bright, sigma)) * exposure
    curves = {"ACES": aces, "Reinhard": reinhard, "Clip": lambda x: np.clip(x, 0, 1)}
    lin = curves[curve](lin)
    out[..., :3] = linear_to_srgb(lin) if encode else lin
    return out'''

PYRAMID_NP = '''\
K5 = np.array([1, 4, 6, 4, 1], dtype=np.float32) / 16


def smooth(a):
    """a, (h, w), blurred by the five weights K5, rows then columns, edges clamped."""
    h, w = a.shape
    p = np.pad(a, ((0, 0), (2, 2)), mode="edge")
    a = sum(K5[i] * p[:, i:i + w] for i in range(5))
    p = np.pad(a, ((2, 2), (0, 0)), mode="edge")
    return sum(K5[i] * p[i:i + h] for i in range(5))


def down(a):
    """Half the size: blurred, then every second pixel."""
    return smooth(a)[::2, ::2]


def up(a, shape):
    """Twice the size, cut to shape: every pixel repeated, then blurred."""
    big = np.repeat(np.repeat(a, 2, axis=0), 2, axis=1)[:shape[0], :shape[1]]
    return smooth(big)


def gaussian_pyramid(a, levels):
    pyramid = [a]
    for _ in range(levels - 1):
        pyramid.append(down(pyramid[-1]))
    return pyramid


def laplacian_pyramid(a, levels):
    g = gaussian_pyramid(a, levels)
    return [g[i] - up(g[i + 1], g[i].shape) for i in range(levels - 1)] + [g[-1]]


def collapse(pyramid):
    """The picture back from its Laplacian pyramid, from the smallest level up."""
    a = pyramid[-1]
    for level in reversed(pyramid[:-1]):
        a = level + up(a, level.shape)
    return a'''

REMAP_GIVEN = '''\
def remap(x, g, sigma_r, alpha, beta):
    """x moved around g: differences under sigma_r as detail, larger ones as an edge."""
    # TODO: t^alpha within sigma_r of g, 1 + beta * (t - 1) beyond, in units of sigma_r
    return x'''

SOLUTION_REMAP = '''\
def remap(x, g, sigma_r, alpha, beta):
    """x moved around g: differences under sigma_r as detail, larger ones as an edge."""
    d = x - g
    t = np.abs(d) / sigma_r
    detail = g + np.sign(d) * sigma_r * t ** alpha
    edge = g + np.sign(d) * sigma_r * (1 + beta * (t - 1))
    return np.where(t <= 1, detail, edge)'''

LOCAL_NP = '''\
def local_laplacian(a, sigma_r, alpha, beta, levels=7):
    """a, (h, w), its small differences scaled as detail and its large ones as edges."""
    gauss = gaussian_pyramid(a, levels)
    refs = np.arange(a.min(), a.max() + 2 * sigma_r, sigma_r)      # the values g, sigma_r apart
    pyramids = [laplacian_pyramid(remap(a, g, sigma_r, alpha, beta), levels) for g in refs]
    out = []
    for i in range(levels - 1):
        # Which two values g this level's Gaussian value lies between, and how far along
        pos = np.clip((gauss[i] - refs[0]) / sigma_r, 0, len(refs) - 1.001)
        k = pos.astype(int)
        w = pos - k
        stack = np.stack([pyramid[i] for pyramid in pyramids])
        rows, cols = np.indices(k.shape)
        out.append((1 - w) * stack[k, rows, cols] + w * stack[k + 1, rows, cols])
    out.append(gauss[-1])
    return collapse(out)'''


GOAL = ("# The tonemapper of section 5, hidden because you write it yourself.\n"
        "def _goal():\n"
        + textwrap.indent("\n\n".join(piece.replace("\n\n\n", "\n\n") for piece in
                                        [HELPERS_NP, CURVES_NP, SOLUTION_AUTO_NP, SOLUTION_APPLY]), "    ")
        + "\n\n    return apply\n\n\n"
        + PARAMS_NP
        + '\n\nsara.live(_goal(), PARAMS, source="HDR", target="Goal")')

DEFAULTS = (f'{{"auto": True, "key": {KEY}, "stops": 0.0, "threshold": {THRESHOLD}, "sigma": {SIGMA}, '
            f'"strength": {STRENGTH}, "curve": "ACES", "encode": True}}')


def answer(snippet, lang="python"):
    """A solution block's code fence."""
    return f"```{lang}\n{snippet}\n```"


# --- Title --------------------------------------------------------------------
md(r"""
# Lekce 6 (bonus): HDR, záře a tone mapping

Fotka ze skutečného světa má světla, která jsou stokrát jasnější než obloha, a obrazovka umí ukázat jen rozsah od 0 do 1. Hry a fotoaparáty proto počítají ve vysokém dynamickém rozsahu (*high dynamic range*, HDR) a teprve na konci světlo stlačí do obrazovky. V této lekci to postavíte celé: expozici, tónovou křivku, automatickou expozici z průměru celého obrázku spočítaného na GPU, záři kolem světel (*bloom*) a stlačení sytých barev v OKLabu. Nepovinná poslední sekce přidá lokální tone mapping.

Lekce je bonusová a na samostudium. Spusťte buňku. Otevře v Sáře dokument s plátnem 640 × 427 a do vrstvy `HDR` zapíše fotku rakety na rampě v lineárním světle, s lampami až stokrát jasnějšími než bílá.
""")

code(f"""
import sys
sys.path.insert(0, "lib")            # sara.py and its helpers live in lib/
import time
import numpy as np
import sara

doc, layer, pixels = sara.init(width={WIDTH}, height={HEIGHT})  # a canvas the size of the photo
hdr = np.load("imgs/6/rocket_hdr.npz")["rgb"].astype(np.float32)   # (h, w, 3), linear light
hdr_layer = doc.new_layer("HDR", np.dstack([hdr, np.ones(hdr.shape[:2], np.float32)]))
print(f"the brightest pixel: {{hdr.max():.0f}}, the mean: {{hdr.mean():.3f}}")
""")

md(r"""
Vrstva drží čísla tak, jak jsou, i větší než 1. Sára je ale ukazuje jako sRGB a všechno nad 1 ořízne, takže vrstva `HDR` vypadá tmavě a lampy jako bílé skvrny.

### Nové funkce v této lekci

Funkce z úvodní lekce shrnuje její příloha. Sloupec Sekce říká, kde se funkce objeví poprvé.

**Kernel v buňce `%%gmacs`**

| Zápis | Co dělá | Sekce |
|---|---|---|
| `exp2(x)`, `log(x)` | $2^x$ a přirozený logaritmus | 1, 2 |
| `{exposure}` na prvním řádku buňky | hodnota proměnné Pythonu, kterou IPython do řádku dosadí | 2 |
| `greaterThanEqual(a, b)`, `all(v)` | porovnání vektorů po složkách a zda platí všude | 4 |
| `-> bool` | funkce, která vrací pravdu nebo nepravdu | 4 |

**Sára a NumPy**

| Volání | Co dělá | Sekce |
|---|---|---|
| `np.load("....npz")["rgb"]` | pole uložené v souboru NumPy | úvod |
| `doc.door.verdict({"reduce": ...})` | součet, průměr, minimum a maximum vrstvy, spočítané na GPU | 2 |
| `a[::2, ::2]` | každý druhý řádek a sloupec, pole poloviční velikosti | 6 |
| `np.repeat(a, 2, axis=0)` | každý řádek dvakrát za sebou | 6 |
| `stack[k, rows, cols]` s `np.indices` | z každého místa hodnota z jiné vrstvy pole, podle pole indexů `k` | 6 |
""")

md(r"""
## Cíl lekce

Buňka níže spustí hotový tonemapper. Jeho kód je schovaný, protože ho během lekce napíšete sami. Zkuste vypnout `auto`, posouvat `stops` a změnit `curve`.
""")

code(GOAL, hidden=True)

# --- 1. Exposure and tone curves -------------------------------------------------
md(r"""
## 1. Expozice a tónová křivka

Fotoaparát světlo nejdřív vynásobí **expozicí**, a ta se měří v **expozičních stupních** (EV, *stops*): o jeden stupeň víc je dvakrát víc světla. Pak ho musí dostat do rozsahu obrazovky. Nejjednodušší je oříznout všechno nad 1. Posuňte `stops` v kernelu níže a podívejte se, co ořezání dělá s lampami a co s oblohou.

@img(exposure_strip.png, 900, Fotka rakety při expozici od minus čtyř do plus čtyř expozičních stupňů, oříznutá na rozsah obrazovky)

Při žádné expozici není vidět obloha a lampy zároveň. Rozsah fotky, od nejtmavšího pixelu po nejjasnější, je asi dvacet expozičních stupňů, obrazovka jich ukáže asi osm.

@img(dynamic_range.png, 900, Histogram logaritmu jasu pixelů fotky, asi dvacet expozičních stupňů, s vyznačeným rozsahem, který ukáže obrazovka)
""")

code(EXPOSURE)

md(r"""
**Tónová křivka** (*tone curve*) místo ořezání světla plynule stlačí: tmavé zůstanou skoro stejné a jasné se přibližují k 1, ale nikdy ji nepřekročí. Reinhardova křivka z roku 2002 je nejjednodušší:

$$T(x) = \frac{x}{1 + x}$$

Filmový standard ACES má křivku ve tvaru písmene S, s tmavšími stíny a měkčím přechodem do bílé. Krzysztof Narkowicz ji v roce 2016 přiblížil zlomkem, který se ve hrách používá dodnes:

$$T(x) = \frac{x\,(2.51\,x + 0.03)}{x\,(2.43\,x + 0.59) + 0.14}$$

@img(tone_curves.png, 760, Ořezání, Reinhardova křivka a ACES na logaritmické ose světla od setiny po sto)

### 🎯 Úkol 1: tónová křivka

Doplňte funkci `tone`: pro `curve` 1 Reinhardovu křivku, pro 2 ACES oříznutou na 0 až 1. Výchozí volba je ACES. Nedokončená funkce ořezává.

<details><summary>💡 Nápověda</summary>

Vzorce platí po složkách, a operace na `vec3` po složkách jsou. `1.0 + x` přičte 1 ke každé složce.
</details>

<details><summary>🔑 Řešení</summary>

""" + answer(SOLUTION_TONE) + r"""
</details>
""")

code(TONE, solve=[(TONE_STUB, SOLUTION_TONE)])

md(r"""
### ✅ Kontrola

Pomocné funkce obsahují převody z lekce o barevných prostorech, jas `luminance` a rozmazání `blur` z lekce o konvoluci. Kontrola porovná vrstvu `Tone` s křivkou ACES v NumPy, s `stops` na 0.
""")

code(HELPERS_NP)

code(CURVES_NP + """


hdr = doc.layer("HDR").read()[..., :3]
sara.check("Tone", rgba(linear_to_srgb(aces(hdr))))
""")

md(r"""
> **❓ Otázka**
> S křivkou ACES vypadá fotka správně až při expozici kolem +1.5. Proč by správná expozice měla záležet na obrázku, a jak by ji šlo spočítat?

<details><summary>🔑 Odpověď</summary>

Tmavá scéna potřebuje víc světla než slunečná, jinak je celá černá, a světlá méně. Fotoaparát expozici volí sám tak, aby průměrný jas scény vyšel jako středně šedá. To je automatická expozice v další sekci.
</details>
""")

# --- 2. Auto exposure ------------------------------------------------------------
md(r"""
## 2. Automatická expozice a redukce na GPU

Expozice má posunout „typický“ jas scény na středně šedou, **klíč** (*key*) kolem 0.18. Typický jas ale není aritmetický průměr: jedna lampa se stonásobným jasem posune průměr víc než celá obloha. Reinhard proto bere průměr **logaritmů**, tedy geometrický průměr:

$$\bar L = \exp\Big(\frac{1}{n} \sum \log(\delta + Y)\Big), \qquad \text{expozice} = \frac{\text{klíč}}{\bar L}$$

Malé $\delta$ zabrání logaritmu nuly. Na logaritmické ose má každý expoziční stupeň stejnou váhu.

Průměr celého obrázku je **redukce**: z milionu čísel jedno. Jedno vlákno by sčítalo pixel po pixelu. GPU sčítá ve stromu: každé vlákno sečte dvojici, pak dvojice součtů, a po $\log_2 n$ krocích zbude jedno číslo. Sára to umí jako krok `reduce`, který vrátí součet, průměr, minimum a maximum každého kanálu vrstvy.

@img(reduce.png, 860, Strom sčítání šestnácti čísel ve čtyřech krocích, v každém kroku polovina vláken sečte dvojici)

Redukce umí jen čísla, která se sčítají v libovolném pořadí. Histogram by potřeboval, aby vlákna zapisovala do společných přihrádek (*atomics*), a ty buňka `%%gmacs` zatím nenabízí.

### 🎯 Úkol 2: logaritmus jasu

Napište kernel, který do všech tří kanálů zapíše $\log(\delta + Y)$, s jasem $Y = 0.2126\,R + 0.7152\,G + 0.0722\,B$. Nedokončený kernel vrací nulu, a expozice pak vyjde rovna klíči.

<details><summary>🔑 Řešení</summary>

""" + answer(SOLUTION_LOG) + r"""
</details>
""")

code(LOG_LUMINANCE, solve=[(LOG_STUB, SOLUTION_LOG)])

md(r"""
Buňka níže pošle Sáře krok `reduce` a z průměru logaritmů spočítá expozici.
""")

code(f"""
def reduce(name):
    \"\"\"Sum, mean, min and max of every channel of the layer, computed on the GPU.\"\"\"
    return doc.door.verdict({{"reduce": {{"layer": name, "ops": ["sum", "mean", "min", "max"]}}}})


said = reduce("Log luminance")
mean_log = said["mean"][0]
key = {KEY}
exposure = key / np.exp(mean_log)
print(f"mean log luminance {{mean_log:.4f}}, exposure {{exposure:.3f}}, {{np.log2(exposure):+.2f}} stops")
""")

md(r"""
### ✅ Kontrola

Průměr spočítaný na GPU porovnáme s NumPy. Vrstva drží čísla v poloviční přesnosti (*half float*), takže se shodují asi na tři místa.
""")

code("""
mean_np = np.log(1e-4 + luminance(hdr)).mean()
print(f"GPU {mean_log:.5f}, NumPy {mean_np:.5f}, difference {abs(mean_log - mean_np):.5f}")
print("they match" if abs(mean_log - mean_np) < 0.01 else "they differ, look at the Log luminance kernel")
""")

md(r"""
Kernel níže použije spočítanou expozici. Zápis `{exposure}` na prvním řádku buňky dosadí hodnotu proměnné `exposure` z Pythonu. Po novém výpočtu expozice spusťte buňku znovu.
""")

code(AUTO)

md(r"""
> **❓ Otázka**
> Zkuste klíč 0.18, 0.12 a 0.05 a buňky spusťte znovu, a nakonec klíč vraťte na 0.12. Proč by hra měnila expozici postupně, snímek po snímku, a ne hned?

<details><summary>🔑 Odpověď</summary>

Klíč říká, jak světlá má scéna vypadat: 0.18 je den, 0.05 noc. Kdyby se expozice měnila okamžitě, obraz by při každém záblesku nebo otočení kamery ke světlu blikal. Hry proto expozici k cíli posouvají pomalu, jako oko, které si na tmu zvyká postupně, zornice za pár sekund, sítnice i desítky minut. Tomu se říká přizpůsobení oka (*eye adaptation*).
</details>
""")

# --- 3. Bloom ---------------------------------------------------------------------
md(r"""
## 3. Záře

Silné světlo se v objektivu i v oku rozptýlí a kolem lampy vznikne záře. Hry ji napodobí třemi kroky: vyberou jen světlo nad prahem, rozmažou ho a přičtou k obrázku **před** tónovou křivkou, dokud je to ještě světlo, ne barva obrazovky.

@img(bloom_steps.png, 900, Obrázek s tónovou křivkou, světlo nad prahem, jeho rozmazání a obrázek se září)

Výběr světla a dva průchody rozmazání z lekce o konvoluci jsou hotové. Rozmazání je bez převodů ze sRGB, protože vrstva drží lineární světlo.
""")

code(BRIGHT)

code(BRIGHT_X)

code(BLOOM)

md(r"""
### 🎯 Úkol 3: složení

Celý kernel čte vrstvu `HDR` jako `src` a záři jako `bloom`. Přičtěte záři vynásobenou `strength` ke světlu, pak expozice a křivka. Nedokončený kernel záři nepřičte.

<details><summary>🔑 Řešení</summary>

""" + answer(SOLUTION_COMPOSITE) + r"""
</details>
""")

code(COMPOSITE, solve=[(COMPOSITE_STUB, SOLUTION_COMPOSITE)])

md(r"""
### ✅ Kontrola
""")

code(f"""
glow = doc.layer("Bloom").read()[..., :3]
sara.check("Glow", rgba(linear_to_srgb(aces((hdr + {STRENGTH} * glow) * exposure))))
""")

md(r"""
> **❓ Otázka**
> Proč se záře přičítá před tónovou křivkou, a ne až k hotovému obrázku?

<details><summary>🔑 Odpověď</summary>

Po tónové křivce jsou lampy i bílá raketa stejně bílé, 1, a práh by je nerozlišil. Před křivkou má lampa sto a raketa jedna, a září jen lampa. Záře přidaná před křivkou je také světlo, a křivka ji stlačí spolu s ostatním. Přidaná po ní by obrázek přesvítila nad 1 a ořízla. Hry rozmazávají pyramidu zmenšených obrázků místo jednoho velkého jádra, protože je to levnější a záře má pak měkký dlouhý ocas.
</details>
""")

# --- 4. Gamut ---------------------------------------------------------------------
md(r"""
## 4. Syté barvy v OKLabu

Křivka po složkách stlačí každý kanál zvlášť. Sytá oranžová lampa má červenou složku mnohem větší než modrou, a křivka červenou stlačí víc. Barva se proto s rostoucím jasem posouvá ke žluté, modrá k azurové, a kde je třetí složka nulová, tam už zůstane. Vývojáři a koloristé tomu říkají „notorious six“: při velkém jasu zbude jen šest barev, tři základní a tři doplňkové.

Druhá možnost je stlačit jen **jas** a barvu vynásobit stejným poměrem. Odstín zůstane, ale jasná sytá barva skončí mimo krychli 0 až 1, kterou obrazovka umí (mimo **gamut**). Ořezání kanálů odstín zase posune. Správně je zachovat v OKLabu světlost a odstín a ubrat **sytost** (*chroma*), dokud se barva do krychle nevejde.

@img(gamut.png, 900, Pruh sytých barev s rostoucí expozicí ve třech podáních: křivka po složkách, jas s ořezáním a jas se stlačením sytosti v OKLabu)

Největší sytost, která se vejde, se najde **půlením intervalu** (*bisection*): podíl sytosti od 0 do 1, a v každém kroku se interval zkrátí na polovinu, podle toho, zda je barva uvnitř. Po 16 krocích je podíl přesný na $2^{-16}$.

### 🎯 Úkol 4: stlačení do gamutu

Doplňte `fit`. Barvu, která je uvnitř, vraťte beze změny. Jinak ji převeďte do OKLabu, světlost ořízněte na 1 a půlením najděte největší podíl sytosti `ab`, při kterém je barva uvnitř. Funkce `inside` je hotová. Nedokončená funkce ořezává, jako volba `Clip`.

<details><summary>💡 Nápověda</summary>

1. `lo = 0.0` je podíl, o kterém víme, že je uvnitř (šedá), `hi = 1.0` ten, který uvnitř není.
2. V cyklu: `mid` uprostřed, barva `vec3(lab.x, mid * lab.yz)` zpět do lineárního RGB, a podle `inside` posuňte `lo` nebo `hi`.
3. Vraťte barvu s podílem `lo`, oříznutou kvůli zaokrouhlení.
</details>

<details><summary>🔑 Řešení</summary>

""" + answer(SOLUTION_FIT) + r"""
</details>
""")

code(GAMUT, solve=[(FIT_STUB, SOLUTION_FIT)])

md(r"""
### ✅ Kontrola

Totéž v NumPy, půlení pro všechny pixely najednou.
""")

code("""
c = hdr * exposure
y = luminance(c)
scaled = c * (aces(y) / np.maximum(y, 1e-4))[..., None]
lab = oklab(scaled)
lab[..., 0] = np.minimum(lab[..., 0], 1.0)
lo, hi = np.zeros(y.shape), np.ones(y.shape)
for i in range(16):
    mid = 0.5 * (lo + hi)
    rgb = oklab_to_linear(np.dstack([lab[..., 0], mid[..., None] * lab[..., 1:]]))
    ok = np.all((rgb >= 0) & (rgb <= 1), axis=-1)
    lo, hi = np.where(ok, mid, lo), np.where(ok, hi, mid)
fitted = np.clip(oklab_to_linear(np.dstack([lab[..., 0], lo[..., None] * lab[..., 1:]])), 0, 1)
inside = np.all((scaled >= 0) & (scaled <= 1), axis=-1)
fitted[inside] = scaled[inside]
sara.check("Gamut", rgba(linear_to_srgb(fitted)))
""")

md(r"""
Přepněte `mode` a porovnejte lampy a oranžový lem u rampy. Rozdíl je jemný, protože fotka má málo sytých světel. Na neonech, ohni nebo laseru ve hře je zřetelný.

> **❓ Otázka**
> Proč se půlí sytost, a ne světlost?

<details><summary>🔑 Odpověď</summary>

Světlost je to, co tónová křivka nastavila, a změna by udělala z jasné lampy tmavší. Odstín je to, co oko pozná nejdřív, a jeho změna je vidět jako jiná barva. Sytost se mění nejméně nápadně: velmi jasná barva i ve skutečnosti vypadá bledší, protože oko se jí přizpůsobí. Proto jasná světla v obrázku přecházejí do bílé, a s OKLabem plynule a bez změny odstínu.
</details>
""")

# --- 5. NumPy and Krita -----------------------------------------------------------
md(r"""
## 5. Do Krity

Tonemapper do Krity spojí všechno v NumPy: automatickou expozici, záři a křivku. Pomocné funkce, `blur` a křivky už máte.

### 🎯 Úkol 5: automatická expozice v NumPy

Napište `auto_exposure(lin, key)` podle vzorce ze sekce 2. Nedokončená funkce vrací 1.

<details><summary>🔑 Řešení</summary>

""" + answer(SOLUTION_AUTO_NP) + r"""
</details>
""")

code(AUTO_NP_GIVEN + """


print(f"NumPy exposure {auto_exposure(hdr):.3f}, GPU exposure {exposure:.3f}")
""", solve=[(AUTO_NP_GIVEN, SOLUTION_AUTO_NP)])

md(r"""
### 🎯 Úkol 6: tonemapper

Napište `apply`:

1. Expozice je $2^{\text{stops}}$, a s `auto` ještě krát `auto_exposure`.
2. Záře: světlo nad prahem `threshold` (jako kernel `Bright`), rozmazané se `sigma`, krát `strength`, přičtené ke světlu.
3. Pak expozice a křivka podle `curve`, a s `encode` převod do sRGB.

Záporné hodnoty na vstupu ořízněte na nulu. Nedokončená funkce vrací obrázek beze změny.

<details><summary>💡 Nápověda</summary>

1. Kernel `Bright` v NumPy: `lin * (np.maximum(y - threshold, 0.0) / np.maximum(y, 1e-4))[..., None]`.
2. Křivku podle jména vybere slovník funkcí: `{"ACES": aces, "Reinhard": reinhard, ...}[curve]`.
</details>

<details><summary>🔑 Řešení</summary>

""" + answer(SOLUTION_APPLY) + r"""
</details>
""")

code(PARAMS_NP + "\n\n\n" + APPLY_GIVEN + """


sara.live(apply, PARAMS, source="HDR", target="Filter NumPy")
""", solve=[(APPLY_GIVEN, SOLUTION_APPLY)])

md(r"""
### ✅ Kontrola

Porovnání s filtrem „Goal“ z úvodu s jeho výchozími hodnotami. Pokud jste s nimi hýbali, spusťte buňku „Goal“ znovu.
""")

code(f"""
defaults = {DEFAULTS}
sara.check("Goal", apply(hdr_layer.read(), **defaults))
""")

md(r"""
### 🎯 Úkol 7: tonemapper v Kritě

Vložte filtr do `pga_filter/effect.py` s `TITLE = "Tonemapper"`. Tonemapper potřebuje obrázek s čísly nad 1: otevřete v Kritě soubor EXR, třeba panorama oblohy z Poly Haven (licence CC0), který má dokument s 32bitovými čísly v lineárním světle. Krita sama převádí lineární světlo na obrazovku, a tak vypněte `encode`.

<details><summary>💡 Nápověda</summary>

1. Do `effect.py` patří pomocné funkce, křivky, `auto_exposure`, `PARAMS` a `apply`.
2. V osmibitovém dokumentu Krita čísla nad 1 ořízne už na vstupu, a tonemapper pak nemá co stlačit.
</details>
""")

# --- 6. Local Laplacian filter (optional) -------------------------------------------
md(r"""
## 6. Lokální kontrast (nepovinné)

Tónová křivka je pro všechny pixely stejná. Stlačí dvacet expozičních stupňů do osmi, a stejně stlačí i malé rozdíly, kresbu mraků nebo nátěr rakety. Obrázek pak vypadá ploše. **Lokální tone mapping** stlačí velké rozdíly mezi oblastmi a malé rozdíly nechá. Sekce je nepovinná a celá v NumPy.

Nejjednodušší pokus rozdělí logaritmus jasu na rozmazaný **základ** a **detail**, rozdíl od základu. Základ stlačí a detail přičte zpět. Rozmazání ale u silné hrany míchá obě strany: kolem jasné lampy je základ jasnější než okolí, a jeho stlačení okolí ztmaví. Vznikne tmavá **svatozář** (*halo*), na obrázku uprostřed.

@img(local_tonemap.png, 900, Výřez s lampami třikrát: s tónovou křivkou pro celý obrázek, se základem a detailem a tmavou svatozáří kolem lamp, a s lokálním Laplaceovým filtrem bez svatozáře)

### Laplaceova pyramida

Burt a Adelson v roce 1983 rozložili obrázek podle velikosti detailů. **Gaussova pyramida** obrázek opakovaně rozmaže malým jádrem $(1\;4\;6\;4\;1) / 16$ a zmenší na polovinu. Každá úroveň **Laplaceovy pyramidy** je rozdíl úrovně Gaussovy pyramidy a zvětšené úrovně pod ní, a poslední úroveň je nejmenší Gaussova:

$$\ell_i = g_i - \text{up}(g_{i+1}), \qquad \ell_n = g_n$$

Rozdíl dvou rozmazání je DoG, tedy skoro Laplaceův operátor z lekce o konvoluci, a odtud jméno. Úroveň $\ell_i$ drží detaily velké asi $2^i$ pixelů. Obrázek se z pyramidy složí zpět přesně: od nejmenší úrovně zvětšit a přičíst další, $g_i = \ell_i + \text{up}(g_{i+1})$.

@img(laplacian_pyramid.png, 860, Gaussova pyramida logaritmu jasu fotky, každá úroveň poloviční, a pod ní Laplaceova pyramida, kde jsou jen hrany a detaily dané velikosti)

Funkce pyramidy jsou hotové. Buňka níže rozloží logaritmus jasu fotky a složí ho zpět.
""")

code(PYRAMID_NP + """


lin = np.maximum(doc.layer("HDR").read()[..., :3], 0.0)
L = np.log2(1e-4 + luminance(lin))                  # log luminance, in stops
pyramid = laplacian_pyramid(L, 7)
print("levels:", [level.shape for level in pyramid])
print(f"collapsed back, largest difference {np.abs(collapse(pyramid) - L).max():.1e}")
""")

md(r"""
### Lokální Laplaceův filtr

Paris, Hasinoff a Kautz v roce 2011 rozhodují o každém koeficientu pyramidy zvlášť. Pro koeficient $\ell_i$ v místě $p$ vezmou hodnotu $g = g_i(p)$ z Gaussovy pyramidy, celý obrázek přemapují funkcí $r_g$ a koeficient vezmou z Laplaceovy pyramidy přemapovaného obrázku. Funkce $r_g$ rozlišuje dva druhy rozdílů od $g$: menší než $\sigma_r$ jsou detail, větší jsou hrana. S $t = |x - g| / \sigma_r$:

$$r_g(x) = g + \operatorname{sign}(x - g)\, \sigma_r \cdot \begin{cases} t^\alpha & t \le 1 \\ 1 + \beta\,(t - 1) & t > 1 \end{cases}$$

S $\alpha < 1$ detail zesílí, s $\beta < 1$ se hrany stlačí. Obě větve se v $t = 1$ potkají, takže $r_g$ je spojitá. Rozdíl přes hranu se nikdy nezesílí, a proto svatozář nevznikne.

@img(remap.png, 700, Funkce r_g kolem g: uvnitř pásu sigma_r detail, beze změny nebo zesílený, vně pásu hrana se sklonem beta)

Přemapovat obrázek zvlášť pro každý koeficient by trvalo dlouho. Rychlá verze (Aubry a kol., 2014) obrázek přemapuje jen pro několik hodnot $g$ s krokem $\sigma_r$ a pro každou postaví Laplaceovu pyramidu. Koeficient v místě $p$ pak lineárně proloží mezi dvěma pyramidami, jejichž $g$ leží kolem $g_i(p)$. Filtr pracuje na logaritmu jasu, kde $\sigma_r = 1$ je jeden expoziční stupeň, a barvu pak vynásobí stejným poměrem jako jas, $2^{\,\text{nový} - \text{starý}}$.

### 🎯 Úkol 8: přemapování

Napište `remap(x, g, sigma_r, alpha, beta)` podle vzorce. `x` je pole, `g` číslo. Funkce `local_laplacian` je hotová. Nedokončená funkce vrací `x`, a filtr pak obrázek nezmění. Kontrola pod funkcí zkusí čtyři hodnoty.

<details><summary>💡 Nápověda</summary>

1. `d = x - g` a `t = np.abs(d) / sigma_r`.
2. Spočítejte obě větve pro celé pole, každou s `np.sign(d)`, a vyberte `np.where(t <= 1, detail, hrana)`.
</details>

<details><summary>🔑 Řešení</summary>

""" + answer(SOLUTION_REMAP) + r"""
</details>
""")

code(REMAP_GIVEN + "\n\n\n" + LOCAL_NP + """


got = remap(np.array([-3.0, -0.5, 0.25, 2.0]), 0.0, 1.0, 0.5, 0.4)
want = np.array([-1.8, -0.7071, 0.5, 1.4])
print("remap: ok" if np.allclose(got, want, atol=1e-3) else f"remap gives {np.round(got, 4)}, expected {want}")
""", solve=[(REMAP_GIVEN, SOLUTION_REMAP)])

md(r"""
Buňka níže zapíše dvě vrstvy: `Global` s automatickou expozicí a křivkou ACES, jako v sekci 2, a `Local`, kde před expozicí a křivkou proběhne lokální Laplaceův filtr. Porovnejte je kolem lamp, na mracích a na raketě.
""")

code(f"""
sigma_r, alpha, beta = {SIGMA_R}, {ALPHA}, {BETA}
start = time.perf_counter()
local = lin * 2.0 ** (local_laplacian(L, sigma_r, alpha, beta) - L)[..., None]
print(f"local Laplacian: {{(time.perf_counter() - start) * 1000:.0f}} ms")
doc.new_layer("Global", rgba(linear_to_srgb(aces(lin * auto_exposure(lin)))))
doc.new_layer("Local", rgba(linear_to_srgb(aces(local * auto_exposure(local)))))
""")

md(r"""
> **❓ Otázka**
> Proč základ a detail udělají kolem lamp svatozář, a lokální Laplaceův filtr ne?

<details><summary>🔑 Odpověď</summary>

Rozmazání neví, kde je hrana. Základ kolem lampy je průměr lampy a tmavého okolí, je tedy jasnější než okolí, a stlačení ho stáhne dolů i s okolím. Lokální filtr se pro každý koeficient ptá, jak daleko je pixel od hodnoty $g$ v tom místě. Lampa je od tmavého okolí dál než $\sigma_r$, a tak je pro okolí hranou: její rozdíl se jen stlačí, nikdy se nepřičte k detailu. Detail okolí, kresba mraků, zůstane, protože je od $g$ blíž než $\sigma_r$.
</details>

> **❓ Otázka**
> Zkuste `alpha` 0.5. Co se stane s oblohou, a proč?

<details><summary>🔑 Odpověď</summary>

Obloha se rozpadne na skvrny. Fotka byla původně osmibitová, a plynulý přechod oblohy se skládá z drobných schodů. Pro filtr jsou to malé rozdíly, tedy detail, a $t^\alpha$ je zesílí nejvíc ze všeho, protože u nuly roste nejstrměji. Je to stejný problém jako šum u druhé derivace. Paris a kol. proto rozdíly menší než úroveň šumu nechávají beze změny. Nakonec vraťte `alpha` na 1.
</details>
""")

# --- Summary ------------------------------------------------------------------
md(r"""
## Shrnutí

- Světlo ve scéně má rozsah i dvacet expozičních stupňů, obrazovka asi osm. Výpočet probíhá v lineárním světle bez omezení a do obrazovky se světlo stlačí až na konci.
- Tónová křivka stlačí jasné plynule. Reinhard je nejjednodušší, ACES má tvar S a filmový vzhled.
- Automatická expozice posune geometrický průměr jasu na klíč. Průměr celého obrázku je redukce, kterou GPU sčítá ve stromu.
- Hodnota z Pythonu se do kernelu dostane jako parametr, dosazením `{proměnná}` do prvního řádku buňky.
- Záře je světlo nad prahem, rozmazané a přičtené před tónovou křivkou.
- Křivka po složkách posouvá odstín jasných barev. Stlačení jasu a pak sytosti v OKLabu odstín zachová.
- Laplaceova pyramida rozloží obrázek podle velikosti detailů a složí ho zpět přesně. Lokální Laplaceův filtr v ní stlačí hrany a detail nechá, bez svatozáře.

### Co jsme vynechali

- **Histogram a percentily.** Expozice podle histogramu ignoruje nejtmavší a nejjasnější procenta pixelů. Histogram na GPU potřebuje atomické zápisy do přihrádek.
- **Lokální Laplaceův filtr na GPU.** Každá úroveň pyramidy je kernel nad menším obrázkem a pyramidy pro všechna $g$ jdou spočítat najednou. Darktable ho má v modulu místního kontrastu.
- **Pyramida záře.** Hry zmenšují obrázek několikrát na polovinu, rozmazávají malé verze a skládají je zpět. Je to rychlejší než velké jádro a záře má dlouhý měkký ocas.
- **Moderní křivky.** AgX a Khronos PBR Neutral řeší posun odstínu přímo v křivce. Blender má obě, Godot zatím jen AgX.
- **HDR obrazovky.** Obrazovka s jasem přes 1000 nitů ukáže víc než 1, a tonemapper pak stlačuje jen to, co se nevejde ani do ní.

### Bonusové úkoly

1. **Postupná adaptace.** V NumPy spočítejte expozici pro sérii snímků se změnou jasu a posouvejte ji k cíli o pár procent na snímek.
2. **Histogram.** Spočítejte v NumPy histogram logaritmu jasu a expozici podle mediánu místo průměru. Jak se liší?
3. **Barevná záře.** Obarvěte záři barvou z výběru `source_color`, převedenou do lineárního světla, nebo ji zesilte jen v jednom kanálu, jako u starých objektivů.
4. **AgX.** Najděte popis křivky AgX a napište ji jako čtvrtou volbu `curve`.
5. **Lokální kontrast v Kritě.** Přidejte do tonemapperu volbu `local` s posuvníky `alpha` a `beta`.
""")

md(r"""
### Nápady na semestrální práci

- Tonemapper do Krity s histogramem v dialogu, lokální expozicí a předvolbami křivek.
- Simulace objektivu: záře, odlesky (*lens flare*), vinětace a chromatická aberace z jedné vrstvy HDR.
- Složení HDR ze tří fotek s různou expozicí (*exposure bracketing*) a jeho tone mapping.
""")


def build():
    lesson_writer.write(cells, OUT, "l6")


if __name__ == "__main__":
    build()
