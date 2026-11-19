"""Builds the lesson 1 notebook in the lecture folder above this one.

The notebook is generated: edit this file and run it again, never the .ipynb
by hand, or the next build overwrites the edit. The pictures, the test
pictures balls.png and color_chart.png and the texture blue_noise.png come from
lesson_01_diagrams.py. The canvas is 512 x 288, the size of the picture Sara
shows under a cell, so the notebook shows every pixel of a dither.

Markdown cells are raw strings, so LaTeX keeps single backslashes and braces.
A picture from imgs/1 is written as @img(file, width, alt text).

Lesson 1 shares its slot with lesson 0, so the class does sections 1 to 3,
two tasks in kernels and a finished kernel that shows when dithering fails, and
the NumPy port with the Krita plugin is homework.
Blue noise is shown as a picture only: a pixel kernel that declares a second
image compiles in Sara but fails to run, as of Sara fb4de39. The goal maps
through three colours, which the class never writes: the homework gets the
three-colour gradient as finished code.

The solutions are constants near the top, so the markdown shows exactly the
code that test_lesson_01.py runs.
"""
import re
from pathlib import Path

import lesson_writer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "lesson_01_gradient_map_dither.ipynb"
IMG = "imgs/1"
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


# --- The kernels and the solutions ------------------------------------------------
# The colours are the hex colours of PARAMS, to four decimals.
COLOURS = """\
uniform vec3 dark: source_color = vec3(0.1059, 0.0627, 0.2078)
uniform vec3 mid: source_color = vec3(0.7608, 0.2549, 0.1765)
uniform vec3 light: source_color = vec3(0.9961, 0.9451, 0.8039)
uniform float mid_at: hint_range(0.05, 0.95) = 0.5"""

# The goal's three colours, dark to mid below mid_at and mid to light above it.
GRADIENT = """\
def gradient(float t) -> vec3:
    if t < mid_at:
        return from_lab(mix(to_lab(dark), to_lab(mid), t / mid_at))
    return from_lab(mix(to_lab(mid), to_lab(light), (t - mid_at) / (1.0 - mid_at)))"""

LAB = """\
def to_lab(vec3 srgb) -> vec3:
    return linear_srgb_to_oklab(srgb_to_linear_srgb(srgb))


def from_lab(vec3 lab) -> vec3:
    return linear_srgb_to_srgb(oklab_to_linear_srgb(lab))"""

BAYER_GMACS = """\
const float BAYER_4X4[16] = float[16](0.0, 8.0, 2.0, 10.0, 12.0, 4.0, 14.0, 6.0, 3.0, 11.0, 1.0, 9.0, 15.0, 7.0, 13.0, 5.0)"""

POSTERISE = """\
def posterise(float t, int n, float d) -> float:
    float levels = float(n - 1)
    return floor(clamp(t, 0.0, 1.0) * levels + d) / levels"""

# The grey of lightness L, shown by the posterise and dither kernels.
GREY_PIXEL = """\
def pixel(ivec2 at) -> vec4:
    vec4 c = src[at]
    float L = posterise(linear_srgb_to_oklab(srgb_to_linear_srgb(c.rgb)).x, steps, threshold(at))
    return vec4(linear_srgb_to_srgb(oklab_to_linear_srgb(vec3(L, 0.0, 0.0))), c.a)"""

BLEND_STUB = """\
        case LINEAR:
            return a              # TODO: mix the amounts of light
        case OKLAB:
            return a              # TODO: mix in OKLab"""

SOLUTION_BLEND = """\
case LINEAR:
    return linear_srgb_to_srgb(mix(srgb_to_linear_srgb(a), srgb_to_linear_srgb(b), t))
case OKLAB:
    vec3 lab = mix(linear_srgb_to_oklab(srgb_to_linear_srgb(a)), linear_srgb_to_oklab(srgb_to_linear_srgb(b)), t)
    return linear_srgb_to_srgb(oklab_to_linear_srgb(lab))"""
BLEND_SOLVED = "\n".join("        " + line for line in SOLUTION_BLEND.splitlines())

DITHER_STUB = """\
def bayer(ivec2 at) -> float:
    # TODO: this pixel's threshold from the matrix, from 0 to 1
    return 0.5


def ign(ivec2 at) -> float:
    # TODO: interleaved gradient noise at this pixel, from 0 to 1
    return 0.5"""

SOLUTION_DITHER = """\
def bayer(ivec2 at) -> float:
    return (BAYER_4X4[(at.y % 4) * 4 + at.x % 4] + 0.5) / 16.0


def ign(ivec2 at) -> float:
    return fract(52.9829189 * fract(dot(vec2(at), vec2(0.06711056, 0.00583715))))"""

# The goal's drop-down. PARAMS lists the same labels with Bayer first, the
# template's default, and the goal opens on Bayer as well.
DITHERS = '"Off", "White noise", "Bayer", "IGN"'

GOAL = f"""\
%%gmacs "balls" -> "Goal"
import linear_srgb_color_space
import oklab_color_space
import noise

{COLOURS}
uniform int steps: hint_range(2, 16) = 5
uniform int dither: hint_enum({DITHERS}) = 2

const int OFF = 0
const int WHITE = 1
const int BAYER = 2
const int IGN = 3
{BAYER_GMACS}


{LAB}


{POSTERISE}


{GRADIENT}


{SOLUTION_DITHER}


def threshold(ivec2 at) -> float:
    match dither:
        case OFF:
            return 0.5
        case WHITE:
            return sa_hash21(vec2(at))
        case BAYER:
            return bayer(at)
        case IGN:
            return ign(at)


def pixel(ivec2 at) -> vec4:
    vec4 c = src[at]
    float t = posterise(to_lab(c.rgb).x, steps, threshold(at))
    return vec4(gradient(t), c.a)"""

# Section 3: a soft light rounded with dither, too late, zoomed, frame by frame and with TAA.
# ZOOM_AT sits on the light's slope, so a large zoom shows dots, not the flat top.
GLOW = f"""\
%%gmacs "balls" -> "Glow"
uniform int steps: hint_range(2, 64) = 16
uniform int dither: hint_enum("Off", "Bayer", "IGN") = 2
uniform bool too_late = false
uniform float zoom: hint_range(0.25, 16) = 1.0
uniform int frame: hint_range(0, 15) = 0
uniform bool taa = false

const int OFF = 0
const int BAYER = 1
const int IGN = 2
{BAYER_GMACS}
const vec2 LIGHT_AT = vec2(256.0, 144.0)
const vec2 ZOOM_AT = vec2(336.0, 144.0)
const int TAA_FRAMES = 16


{POSTERISE}


def threshold(ivec2 at, int f) -> float:
    match dither:
        case OFF:
            return 0.5
        case BAYER:
            ivec2 b = at + ivec2(f % 4, f / 4)                    # the matrix moves every frame
            return (BAYER_4X4[(b.y % 4) * 4 + b.x % 4] + 0.5) / 16.0
        case IGN:
            vec2 q = vec2(at) + 5.588238 * float(f)               # the noise moves every frame
            return fract(52.9829189 * fract(dot(q, vec2(0.06711056, 0.00583715))))


def light(ivec2 at) -> float:
    \"\"\"A soft light on black, in the screen's sRGB values, before any rounding.\"\"\"
    vec2 d = (vec2(at) - LIGHT_AT) / 112.0
    return 0.5 * exp(-dot(d, d))


def rounded(ivec2 at, int f) -> float:
    \"\"\"The light rounded to steps levels in frame f.\"\"\"
    float v = light(at)
    if too_late:
        v = posterise(v, steps, 0.5)                              # rounded already, as in an 8-bit texture
    return posterise(v, steps, threshold(at, f))


def shown(ivec2 at) -> float:
    \"\"\"One frame, or with TAA the mean of sixteen frames in a row.\"\"\"
    if not taa:
        return rounded(at, frame)
    float total = 0.0
    for k in range(TAA_FRAMES):
        total += rounded(at, frame + k)
    return total / float(TAA_FRAMES)


def pixel(ivec2 at) -> vec4:
    vec2 p = (vec2(at) + 0.5 - ZOOM_AT) / zoom + ZOOM_AT       # the finished picture, zoomed around ZOOM_AT
    return vec4(vec3(shown(ivec2(floor(p)))), 1.0)"""

HELPERS_NP = '''\
def fract(x):
    """Same as fract in GLSL: the part after the decimal point."""
    return x - np.floor(x)


def mix(a, b, t):
    """Same as mix in GLSL: (1 - t) * a + t * b."""
    return (1.0 - t) * a + t * b


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


def to_lab(srgb):
    """sRGB colours, one or a whole array of them, to OKLab."""
    lms = srgb_to_linear(np.asarray(srgb, dtype=np.float32)) @ M1.T
    return np.cbrt(lms) @ M2.T


def from_lab(lab):
    """OKLab back to sRGB."""
    lms = (lab @ M2_INV.T) ** 3
    return linear_to_srgb(lms @ M1_INV.T)


def gradient(t, dark, mid, light, mid_at):
    """Colours of shape (h, w, 3) for lightness t of shape (h, w), dark through mid to light in OKLab."""
    stops = to_lab([dark, mid, light])               # one OKLab colour per row
    lab = np.empty(t.shape + (3,), dtype=np.float32)
    for k in range(3):                               # L, a and b, each on its own
        lab[..., k] = np.interp(t, [0.0, mid_at, 1.0], stops[:, k])
    return from_lab(lab)


def hash21(x, y):
    """White noise from 0 to 1 at pixel positions x, y, the same as sa_hash21 in a kernel."""
    q = fract(np.stack([x, y, x], axis=-1).astype(np.float32) * np.float32(0.1031))
    q += (q * (q[..., [1, 2, 0]] + np.float32(33.33))).sum(axis=-1, keepdims=True)
    return fract((q[..., 0] + q[..., 1]) * q[..., 2])


BAYER = np.array([
    [0, 8, 2, 10],
    [12, 4, 14, 6],
    [3, 11, 1, 9],
    [15, 7, 13, 5],
], dtype=np.float32)

PARAMS = {
    "dark": "#1b1035",
    "mid": "#c2412d",
    "light": "#fef1cd",
    "mid_at": (0.5, 0.05, 0.95),
    "steps": (5, 2, 16),
    "dither": ["Bayer", "IGN", "White noise", "Off"],
}'''

APPLY_NP_GIVEN = '''\
def apply(img, dark, mid, light, mid_at, steps, dither):
    out = img.copy()
    # TODO: lightness, threshold, posterise, then the colours from gradient
    return out'''

SOLUTION_APPLY_NP = '''\
def apply(img, dark, mid, light, mid_at, steps, dither):
    out = img.copy()
    t = to_lab(img[..., :3])[..., 0]                  # the lightness of every pixel
    yy, xx = np.mgrid[0:t.shape[0], 0:t.shape[1]]
    if dither == "Bayer":
        d = (BAYER[yy % 4, xx % 4] + 0.5) / 16
    elif dither == "IGN":
        d = fract(52.9829189 * fract(0.06711056 * xx + 0.00583715 * yy))
    elif dither == "White noise":
        d = hash21(xx, yy)
    else:
        d = 0.5
    levels = steps - 1
    t = np.floor(np.clip(t, 0.0, 1.0) * levels + d) / levels
    out[..., :3] = gradient(t, dark, mid, light, mid_at)
    return out'''


def answer(snippet, lang="python"):
    """A solution block's code fence."""
    return f"```{lang}\n{snippet}\n```"


# --- Title --------------------------------------------------------------------
md(r"""
# Lekce 1: Gradientní mapa s ditheringem

Obrázek přebarvíte podle světlosti přechodem několika barev, omezíte ho na pár odstínů a pruhy mezi nimi rozbijete šumem.
""")

code("""
import sys
sys.path.insert(0, "lib")            # sara.py and its helpers live in lib/
import numpy as np
import sara

doc, layer, pixels = sara.init(width=512, height=288) # a new sRGB document in Sara
chart = doc.import_image("imgs/1/color_chart.png")    # a layer called "color_chart"
balls = doc.import_image("imgs/1/balls.png")          # a layer called "balls"
balls.show()
""")

md(r"""
### Nové funkce v této lekci

**Kernel v buňce `%%gmacs`**

| Zápis | Co dělá | Sekce |
|---|---|---|
| `uniform vec3 c: source_color = vec3(1.0, 0.5, 0.2)` | výběr barvy, kernel dostane barvu v sRGB | 1 |
| `uniform int i: hint_enum("A", "B") = 0` | rozbalovací seznam, kernel dostane pořadí volby od 0 | 1 |
| `const int B = 1` | konstanta, tady jméno pro jednu volbu ze seznamu | 1 |
| `match i:` a pod ním `case B:` | větvení podle hodnoty, `switch` z GLSL | 1 |
| `fract(x)`, `clamp(x, 0.0, 1.0)` | desetinná část, omezení do intervalu | 2 |
| `sa_hash21(vec2 p)` | bílý šum, číslo od 0 do 1 z pozice, knihovna `noise` | 2 |
| `const float A[16] = float[16](...)`, `at.x % 4` | pole konstant, zbytek po dělení | 2 |
| `uniform bool b = false` | zaškrtávací políčko pod buňkou | 3 |
| `exp(x)` | $e^x$ | 3 |

**Sára v Pythonu**

| Volání | Co dělá | Sekce |
|---|---|---|
| `sara.live(apply, PARAMS, source="balls", target="...")` | jednou přečte vrstvu `source`, z `PARAMS` udělá ovládací prvky a po každé změně zapíše `apply(obrázek, **hodnoty)` do vrstvy `target` | 4 |

**NumPy**

| Volání | Co dělá | Sekce |
|---|---|---|
| `BAYER[yy % 4, xx % 4]` | indexování polem: každému pixelu vybere prvek matice 4 × 4 | 4 |
| `np.interp(t, xp, fp)` | po částech lineární funkce: v bodech `xp` má hodnoty `fp`, mezi nimi interpoluje | 4 |
| `np.floor(x)`, `np.clip(x, 0, 1)` | totéž co `floor` a `clamp` v kernelu, po prvcích | 4 |
""")

md(r"""
## Cíl lekce
""")

code(GOAL, hidden=True)

# --- 1. Two colours ----------------------------------------------------------------
md(r"""
## 1. Dvě barvy místo šedé

Gradientní mapa (*gradient map*) přebarví obrázek podle světlosti: stíny jednou barvou, světla druhou a mezi nimi plynulý přechod, který působí rovnoměrně. Se dvěma barvami se jí říká duotone. Světlost pixelu $t$ od 0 do 1 už umíte spočítat, je to $L$ z OKLabu, takže stačí obě barvy smíchat podle $t$. Zkuste `mix(dark, light, t)` s modrou a žlutou a podívejte se na střed šedotónu (*grayscale*) dole:
""")

code("""
%%gmacs "color_chart" -> "Duotone"
import linear_srgb_color_space
import oklab_color_space

uniform vec3 dark: source_color = vec3(0.0, 0.0, 1.0)
uniform vec3 light: source_color = vec3(1.0, 1.0, 0.0)


def pixel(ivec2 at) -> vec4:
    vec4 c = src[at]
    float t = linear_srgb_to_oklab(srgb_to_linear_srgb(c.rgb)).x    # lightness L from 0 to 1
    return vec4(mix(dark, light, t), c.a)
""")

md(r"""
Uprostřed je šedá, přechod rovnoměrně nepůsobí. `mix` tu míchá čísla sRGB, a ta nejsou ani množství světla, ani barva, jak ji vidí oko. Barvy jde míchat i jinde:

@img(mix_spaces.png, 900, Tři přechody namíchané v sRGB, v lineárním RGB a v OKLabu)

| Prostor | Co jsou čísla | K čemu |
|---|---|---|
| sRGB | hodnoty, jak je ukládá obrázek | k uložení, ne k počítání |
| lineární RGB | množství světla | ke sčítání světla: rozostření, zmenšení, průhlednost |
| OKLab | barva, jak ji vnímá oko | k přechodům a paletám |

### 🎯 Úkol 1: míchání v lineárním RGB a v OKLabu

Doplňte ve funkci `blend` větve `LINEAR` a `OKLAB`, přepínejte `space` a zkuste i jiné dvojice barev. Barvy `a`, `b` i výsledek jsou v sRGB.

<details><summary>💡 Nápověda</summary>

1. Převody jsou `srgb_to_linear_srgb`, `linear_srgb_to_srgb`, `linear_srgb_to_oklab` a `oklab_to_linear_srgb`. Do OKLabu se jde přes lineární RGB.
2. Převeďte obě barvy, smíchejte je a výsledek převeďte zpátky do sRGB.
</details>

<details><summary>🔑 Řešení</summary>

""" + answer(SOLUTION_BLEND) + r"""
</details>
""")

code("""
%%gmacs "color_chart" -> "Duotone"
import linear_srgb_color_space
import oklab_color_space

uniform vec3 dark: source_color = vec3(0.0, 0.0, 1.0)
uniform vec3 light: source_color = vec3(1.0, 1.0, 0.0)
uniform int space: hint_enum("sRGB", "Linear", "OKLab") = 0

const int SRGB = 0
const int LINEAR = 1
const int OKLAB = 2


def blend(vec3 a, vec3 b, float t) -> vec3:
    match space:
        case LINEAR:
            return a              # TODO: mix the amounts of light
        case OKLAB:
            return a              # TODO: mix in OKLab
    return mix(a, b, t)           # SRGB


def pixel(ivec2 at) -> vec4:
    vec4 c = src[at]
    float t = linear_srgb_to_oklab(srgb_to_linear_srgb(c.rgb)).x
    return vec4(blend(dark, light, t), c.a)
""", solve=[(BLEND_STUB, BLEND_SOLVED)])

md(r"""
V OKLabu je střed šedotónu světle modrozelený, v lineárním RGB světle šedý.

> **❓ Otázka**
> Ve kterém prostoru zabírají tmavé modré odstíny nejkratší úsek šedotónu a proč?

<details><summary>🔑 Odpověď</summary>

V lineárním RGB. V polovině přechodu je tam polovina světla a ta oku připadá světlá: šedá s polovinou světla má světlost asi 0.8. Tmavé odstíny se proto stlačí na začátek. OKLab dělí přechod podle vnímané světlosti.
</details>
""")

# --- 2. Posterise and dither ----------------------------------------------------
md(r"""
## 2. Pár odstínů bez pruhů

Retro hry měly jen pár odstínů, Game Boy čtyři. Světlost zaokrouhlíme na $n$ úrovní:

$$q = \frac{\lfloor t\,(n - 1) + d \rfloor}{n - 1}, \qquad d = 0.5$$

Tomu se říká posterizace (*posterization*). Zkuste `steps` od 2 do 8 a sledujte stínování míčků:
""")

code(f"""
%%gmacs "balls" -> "Dither"
import linear_srgb_color_space
import oklab_color_space
import noise

uniform int steps: hint_range(2, 16) = 4
uniform int dither: hint_enum("Off", "White noise") = 0

const int OFF = 0
const int WHITE = 1


{POSTERISE}


def threshold(ivec2 at) -> float:
    match dither:
        case OFF:
            return 0.5
        case WHITE:
            return sa_hash21(vec2(at))


{GREY_PIXEL}
""")

md(r"""
Stínování se rozpadlo na ploché pruhy (*banding*). Víc úrovní (větší bitovou hloubku, *bit depth*) si ale vybrat nemůžeme, určuje je výstup. Game Boy měl čtyři odstíny, GIF nejvýš 256 barev, tisk a e-ink jen dva, inkoust a papír. I běžný monitor s 256 úrovněmi na kanál ukáže pruhy v pomalém tmavém přechodu, třeba na obloze nebo kolem světla na černém pozadí.

Chceme docílit tohoto:

@img(dither_goal.png, 512, Míčky na čtyřech odstínech šedé, nahoře zaokrouhlené, dole s ditheringem)

Dole jsou pořád jen čtyři odstíny, a přece z dálky vypadá stínování plynule. Oko totiž sousední pixely zprůměruje. Pixelů je dost, tak část rozlišení vyměníme za odstíny, které výstupu chybějí.

Jak na to? Vezměte plochu, jejíž světlost leží ve třech čtvrtinách cesty k vyšší úrovni. S prahem $d = 0.5$ se celá zaokrouhlí nahoru, a jiné $d$, stejné pro všechny pixely, jen posune hranici pruhu. Aby měla plocha z dálky správnou světlost, musí se nahoru zaokrouhlit tři pixely ze čtyř a dolů jeden. Práh (*threshold*) $d$ se tedy musí měnit od pixelu k pixelu.

Zkusíme to třeba takhle: každý pixel dostane náhodné $d$ od 0 do 1, bílý šum (*white noise*). Přepněte v buňce nahoře `dither` na `White noise`.

A ono to funguje. Pruhy zmizely a stínování je z dálky zase plynulé. Tomu se říká dithering.

Zblízka je ale šum zrnitý, protože se náhodná čísla shlukují. Od dobrého ditheringu proto chceme tři věci:

1. Plocha má z dálky stejnou světlost jako před zaokrouhlením, takže pruhy zmizí.
2. Šum, který to způsobí, je vidět co nejméně.
3. Každý pixel se spočítá sám, bez čekání na sousedy, jak to potřebuje GPU.

Bílý šum splní první a třetí. Šum bez shluků, ve kterém se sousední prahy vždy hodně liší, se jmenuje modrý šum (*blue noise*) a splní všechny tři. Počítá se ale pomalu, proto se předem uloží do textury, třeba 64 × 64 pixelů, a ta se opakuje přes celý obrázek (obrázek níže, prostřední sloupec).

Jde to i bez textury:

- **Bayerova matice** 4 × 4 je šestnáct prahů rozložených tak, aby se sousedé co nejvíc lišili. Opakuje se po čtyřech pixelech, práh pixelu je její prvek v řádku $y \bmod 4$ a sloupci $x \bmod 4$:

$$B = \begin{pmatrix} 0 & 8 & 2 & 10 \\ 12 & 4 & 14 & 6 \\ 3 & 11 & 1 & 9 \\ 15 & 7 & 13 & 5 \end{pmatrix}, \qquad d(x, y) = \frac{B_{y \bmod 4,\; x \bmod 4} + 0.5}{16}$$

Přičtená polovina posune prahy doprostřed jejich šestnáctin, od 1/32 do 31/32, takže jejich průměr je 0.5 jako u prostého zaokrouhlení. V kernelu je matice hotová jako pole `BAYER_4X4`. Její pravidelný vzor je vidět, druhou podmínku tedy splní jen napůl, zato se hodí pro retro vzhled.

- **Interleaved gradient noise** (IGN) je jeden vzorec, který dá sousedům rozdílné prahy bez opakující se mřížky (Jorge Jimenez, Call of Duty: Advanced Warfare, 2014):

$$\mathrm{IGN}(x, y) = \operatorname{fract}\bigl(52.9829189 \cdot \operatorname{fract}(0.06711056\,x + 0.00583715\,y)\bigr)$$

@img(bayer.png, 900, Bayerova matice 4 × 4 a přechod převedený na dvě úrovně bez ditheringu a s Bayerovým prahem)

@img(noise.png, 1000, Pět prahů a měkké světlo na černém pozadí zaokrouhlené na osm úrovní s každým z nich)

> **👉 Tip**
> Acerola ve videu [Reverse Engineering The Graphics Of Silksong](https://www.youtube.com/watch?v=au9pce-xg5s&t=960s) (anglicky, dithering od 16. minuty) ukazuje pruhy kolem světel na tmavém pozadí, proč je Bayer neskryje a jak je IGN odstraní jedním řádkem shaderu.

### 🎯 Úkol 2: Bayer a IGN

Napište `bayer(at)` a `ign(at)` podle vzorců výše. Pak porovnejte všechny čtyři volby se `steps` 2 a 8. Nedokončené funkce vracejí 0.5 jako `Off`.

<details><summary>💡 Nápověda</summary>

1. `BAYER_4X4` je jedno pole o 16 číslech řádek po řádku, prvek v řádku `j` a sloupci `i` má index `j * 4 + i`. Zbytek po dělení je `%`.
2. Pro IGN převeďte pozici na desetinná čísla, `vec2(at)`. Součet $0.067\,x + 0.0058\,y$ je skalární součin, `dot`.
</details>

<details><summary>🔑 Řešení</summary>

""" + answer(SOLUTION_DITHER) + r"""
</details>
""")

code(f"""
%%gmacs "balls" -> "Dither"
import linear_srgb_color_space
import oklab_color_space
import noise

uniform int steps: hint_range(2, 16) = 4
uniform int dither: hint_enum({DITHERS}) = 2

const int OFF = 0
const int WHITE = 1
const int BAYER = 2
const int IGN = 3
{BAYER_GMACS}


{POSTERISE}


{DITHER_STUB}


def threshold(ivec2 at) -> float:
    match dither:
        case OFF:
            return 0.5
        case WHITE:
            return sa_hash21(vec2(at))
        case BAYER:
            return bayer(at)
        case IGN:
            return ign(at)


{GREY_PIXEL}
""", solve=[(DITHER_STUB, SOLUTION_DITHER)])

md(r"""
Se `steps = 4` rozpadne Bayer hranice mezi odstíny na pravidelný vzor a IGN na jemný šum bez mřížky. Z dálky vypadá stínování zase plynule.

@img(floyd_steinberg.png, 620, Chyba zaokrouhleného pixelu se rozdělí čtyřem sousedům s vahami 7/16, 3/16, 5/16 a 1/16)

> **❓ Otázka**
> Nejznámější dithering, Floyd-Steinberg, prochází pixely po řádcích, každý zaokrouhlí a chybu rozdělí sousedům, kteří přijdou na řadu později. Kterou ze tří podmínek nesplní?

<details><summary>🔑 Odpověď</summary>

Třetí. Pixel se smí zaokrouhlit až po přičtení chyb od souseda vlevo a od tří sousedů nad ním, a ti čekají na své sousedy. Vlákna by na sebe čekala jedno po druhém. Bílý šum, modrý šum, Bayer i IGN závisí jen na pozici pixelu, takže si každé vlákno spočítá práh samo.
</details>
""")

# --- 3. When dithering fails ------------------------------------------------------
md(r"""
## 3. Kdy dithering pomůže a kdy ne

Světlo na tmavém pozadí, třeba lampa v noci, je ve hrách častý zdroj pruhů. Slábne pomalu přes mnoho pixelů, takže každý odstín zabere široký pruh. Kernel níže kreslí takové měkké světlo a s jeho ovládacími prvky vyzkoušíte, kdy dithering pomůže a kdy ne.
""")

code(GLOW)

md(r"""
- **Pozdě.** Zaškrtněte `too_late`. Světlo se pak nejdřív zaokrouhlí bez šumu, jako když je uložené v 8bitové textuře nebo už vykreslené na obrazovce, a dithering přijde až potom. Pruhy zůstanou, protože šum kolem už zaokrouhlené hodnoty se na ni zase zaokrouhlí. Tak dopadne dithering přidaný až na hotový obrázek jako efekt po vykreslení (*post-processing*). Dithering patří do chvíle, kdy se zaokrouhluje, tedy do shaderu, který světlo kreslí.
- **Zvětšení a zmenšení.** Posuňte `zoom`, kernel pak ukáže hotový obrázek zvětšený nebo zmenšený, jako texturu ve hře nebo fotku na sociální síti. Zvětšený vzor je vidět stejně jako pruhy. Při zmenšení na čtvrtinu s Bayerem připadne každému ukázanému pixelu stejný práh matice a pruhy se vrátí. Dithering proto patří až na konec, v rozlišení, ve kterém se obrázek ukáže. I tahle lekce má proto plátno jen 512 × 288 pixelů, stejně velké jako obrázek pod buňkou, který tak ukáže každý pixel. Pokazí ho i komprese JPEG nebo videa.
- **V čase.** Nastavte `zoom` na 16, aby byly vidět jednotlivé pixely, a táhněte posuvníkem `frame`, který snímek se ukáže. Prahy se každým snímkem posunou a šum bliká. Pak zaškrtněte `taa`: kernel zprůměruje 16 snímků po sobě, jako to ve hrách dělá časový antialiasing (*temporal anti-aliasing*, TAA). Obraz se při tažení `frame` přestane hýbat a ze šumu se stanou mezistupně, které výstup nemá. Právě pro TAA vznikl IGN. Bez TAA je lepší šum, který se mezi snímky nemění.
""")

# --- 4. NumPy and Krita, at home --------------------------------------------------
md(r"""
## 4. Doma: do Krity

`PARAMS` znáte z úvodní lekce. Nové jsou v něm dva druhy položek:

| V `PARAMS` | Ovládací prvek | `apply` dostane |
|---|---|---|
| `"dark": "#1b1035"` | výběr barvy | trojici čísel `(r, g, b)` od 0 do 1 v sRGB |
| `"dither": ["Bayer", "IGN"]` | rozbalovací seznam, první volba je výchozí | text vybrané volby, třeba `"IGN"` |

Pomocné funkce znáte z úvodní lekce. Nová je `hash21`, bílý šum stejný jako `sa_hash21` v kernelu, a hotová `gradient`, přechod tří barev z cíle lekce. Pod `mid_at` míchá `dark` a `mid`, nad ním `mid` a `light`, obojí v OKLabu. Takovou po částech lineární funkci spočítá `np.interp(t, xp, fp)` pro celé pole najednou, pro každou složku OKLabu zvlášť.
""")

code(HELPERS_NP)

md(r"""
### 🎯 Úkol 3: efekt v NumPy

Napište `apply`: světlost, práh podle `dither`, posterizace na `steps` úrovní a nakonec barvy z hotové `gradient`.

<details><summary>💡 Nápověda</summary>

1. Světlost všech pixelů je `to_lab(img[..., :3])[..., 0]`.
2. Souřadnice pixelů dá `np.mgrid` jako v úvodní lekci. `BAYER[yy % 4, xx % 4]` je práh pro všechny pixely najednou.
3. IGN napíšete s `fract` jako v kernelu, jen místo `dot` sečtete dva součiny. Bílý šum je `hash21(xx, yy)`, bez ditheringu `d = 0.5`.
4. Barvy všech pixelů vrátí `gradient(t, dark, mid, light, mid_at)` jako pole tvaru (výška, šířka, 3).
</details>

<details><summary>🔑 Řešení</summary>

""" + answer(SOLUTION_APPLY_NP) + r"""
</details>
""")

code(APPLY_NP_GIVEN + """


sara.live(apply, PARAMS, source="balls", target="Goal NumPy")
""", solve=[(APPLY_NP_GIVEN, SOLUTION_APPLY_NP)])

md(r"""
### 🎯 Úkol 4: gradientní mapa v Kritě

Vložte efekt do `pga_filter/effect.py` a vyzkoušejte ho na vlastní fotce. Okno filtru ukáže tři výběry barvy, dva posuvníky a rozbalovací seznam.

<details><summary>💡 Nápověda</summary>

1. Do `effect.py` patří buňka s pomocnými funkcemi a `PARAMS` a vaše `apply`.
2. `TITLE = "Gradient map"` pojmenuje položku v menu.
</details>
""")

# --- Summary ------------------------------------------------------------------
md(r"""
## Shrnutí

- Hodnoty sRGB slouží k uložení. Světlo se sčítá v lineárním RGB, přechody se míchají v OKLabu.
- Posterizace zaokrouhlí na $n$ úrovní a nechá pruhy. Dithering je rozbije prahem, který se mění od pixelu k pixelu.
- Bílý šum se shlukuje, modrý ne, ale potřebuje texturu. Bayer je malá textura se vzorem, IGN jeden vzorec bez vzoru.
- Floyd-Steinberg šíří chybu na sousedy, a pixely na sebe proto musí čekat.
- Dithering patří do chvíle zaokrouhlení a na konec, v rozlišení obrazovky. Už zaokrouhlený, zvětšený ani zmenšený obrázek nezachrání. S TAA se šum měnící se po snímcích zprůměruje na mezistupně.

### Co jsme vynechali

- **Barvy mimo gamut.** Přechod v OKLabu může projít barvou, kterou sRGB neumí zobrazit, a ta se ořízne. Mapování gamutu přijde v lekci o HDR.
- **Jak vzniká modrý šum.** Textura je spočítaná algoritmem void-and-cluster (Ulichney, 1993) ve skriptu `build/lesson_01_diagrams.py`.

### Bonusové úkoly

1. **Modrý šum.** Textura je v `imgs/1/blue_noise.png`. V notebooku ji načte `plt.imread(...)[..., 0]` a práh je `BLUE[yy % 64, xx % 64]` jako u Bayera. Pro Kritu ji uložte vedle `effect.py` přes `np.save` a v `apply` načtěte `np.load(Path(__file__).with_name("blue_noise.npy"))`.
2. **Bayer 8 × 8.** Postaví se ze 4 × 4: čtyři kopie $4B$ s přičtenými 0, 2, 3 a 1 v rozích (levý horní, pravý horní, levý dolní, pravý dolní).
3. **Retro paleta.** Posterizujte každý kanál R, G, B zvlášť na dvě úrovně, s ditheringem i bez. Kolik barev výsledek má?

### Nápady na semestrální práci

- Gradientní mapa s předvolbami palet. Seznam v `PARAMS` jako `["Sunset", "Ocean", "Game Boy"]` vybere sadu barev, výběry barvy ji doladí.
- Retro filtr s pevnou paletou a ditheringem. Matice prahů může být parametr, v `PARAMS` jako `[[0, 2], [3, 1]]` se z ní stane mřížka čísel k úpravě.
""")


def build():
    lesson_writer.write(cells, OUT, "l1")

if __name__ == "__main__":
    build()
