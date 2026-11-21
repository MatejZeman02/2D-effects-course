"""Builds the lesson 2 notebook in the lecture folder above this one.

The notebook is generated: edit this file and run it again, never the .ipynb
by hand, or the next build overwrites the edit. The pictures come from
lesson_02_diagrams.py.

Markdown cells are raw strings, so LaTeX keeps single backslashes and braces.
A picture from imgs/2 is written as @img(file, width, alt text).

The solutions are constants near the top, so the markdown shows exactly the
code that test_lesson_02.py runs.
"""
import re
from pathlib import Path

import lesson_writer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "lesson_02_noise_terrain.ipynb"
IMG = "imgs/2"
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


# --- The terrain's colour stops ---------------------------------------------------
# Height above the sea level, an sRGB colour, the Czech label of the diagram
# and the English name of the code comment. The kernel gets them in OKLab,
# computed here, and NumPy converts the same hex colours with its own to_lab.
STOPS = [
    (-0.20, "#0b2350", "hluboká voda", "deep water"),
    (-0.02, "#2b6ca3", "voda", "water"),
    (0.00, "#5fa8c8", "mělčina", "shallows"),
    (0.008, "#cbb27a", "písek", "sand"),
    (0.03, "#6f9e3f", "tráva", "grass"),
    (0.10, "#3e6b2c", "les", "forest"),
    (0.17, "#7a6a58", "skála", "rock"),
    (0.25, "#f2f2f0", "sníh", "snow"),
]


def _srgb_to_oklab(text):
    """One hex colour to OKLab, the same formulas as the notebooks' to_lab."""
    rgb = [int(text[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    m1 = [[0.4122214708, 0.5363325363, 0.0514459929],
          [0.2119034982, 0.6806995451, 0.1073969566],
          [0.0883024619, 0.2817188376, 0.6299787005]]
    m2 = [[0.2104542553, 0.7936177850, -0.0040720468],
          [1.9779984951, -2.4285922050, 0.4505937099],
          [0.0259040371, 0.7827717662, -0.8086757660]]
    lms = [sum(m1[r][k] * lin[k] for k in range(3)) ** (1 / 3) for r in range(3)]
    return [sum(m2[r][k] * lms[k] for k in range(3)) for r in range(3)]


def _float(x):
    """A GLSL float literal, always with a decimal point."""
    text = f"{x:g}"
    return text if "." in text else text + ".0"


STOPS_GMACS = (
    f"const float STOP_AT[{len(STOPS)}] = float[{len(STOPS)}]("
    + ", ".join(_float(at) for at, _, _, _ in STOPS) + ")\n"
    + f"const vec3 STOP_LAB[{len(STOPS)}] = vec3[{len(STOPS)}](\n"
    + ",\n".join("    vec3(" + ", ".join(f"{v:.4f}" for v in _srgb_to_oklab(hex_)) + ")" for _, hex_, _, _ in STOPS)
    + ")"
)

# --- The kernels and the solutions ------------------------------------------------
HASH_GMACS = """\
# One 32-bit number from a lattice cell: the bits of x and y mixed until
# neighbouring cells share nothing. Integers only, so the GPU and NumPy agree.
def cell_hash(ivec2 cell) -> uint:
    uint h = uint(cell.x) * 1597334677u + uint(cell.y) * 3812015801u
    h = (h ^ (h >> 16u)) * 2146121005u
    h = (h ^ (h >> 15u)) * 2221713035u
    return h ^ (h >> 16u)


# A number from 0 to 1 for a cell, from the hash's top 24 bits.
def random1(ivec2 cell) -> float:
    return float(cell_hash(cell) >> 8u) / 16777216.0


# Two numbers from 0 to 1 for a cell, from the hash's two halves.
def random2(ivec2 cell) -> vec2:
    uint h = cell_hash(cell)
    return vec2(float(h & 65535u), float(h >> 16u)) / 65536.0"""

SOLUTION_VALUE = """\
def value_noise(vec2 p, bool smoothed) -> float:
    ivec2 cell = ivec2(floor(p))
    vec2 f = fract(p)
    if smoothed:
        f = f * f * (3.0 - 2.0 * f)
    float a = random1(cell)
    float b = random1(cell + ivec2(1, 0))
    float c = random1(cell + ivec2(0, 1))
    float d = random1(cell + ivec2(1, 1))
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y)"""

# The stubs of tasks 1 and 2, which the solved copy replaces.
VALUE_STUB = """\
def value_noise(vec2 p, bool smoothed) -> float:
    # TODO: the random1 of the cell's four corners, mixed by the position inside it
    return random1(ivec2(floor(p)))"""

FBM_STUB = """\
def fbm(vec2 p) -> float:
    # TODO: octaves layers of value_noise, each lacunarity times finer and gain times weaker
    return value_noise(p)"""

# Task 1's solution with the smooth curve always on, given from task 2 on.
VALUE_GMACS = """\
def value_noise(vec2 p) -> float:
    ivec2 cell = ivec2(floor(p))
    vec2 f = fract(p)
    f = f * f * (3.0 - 2.0 * f)
    float a = random1(cell)
    float b = random1(cell + ivec2(1, 0))
    float c = random1(cell + ivec2(0, 1))
    float d = random1(cell + ivec2(1, 1))
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y)"""

SOLUTION_FBM = """\
def fbm(vec2 p) -> float:
    float total = 0.0
    float weight = 0.0
    float amplitude = 1.0
    for i in range(octaves):
        total += amplitude * value_noise(p)
        weight += amplitude
        amplitude *= gain
        p = p * lacunarity + 0.5
    return total / weight"""

# Task 2's solution at gain 0.5 and lacunarity 2, given in the terrain.
FBM_GMACS = """\
def fbm(vec2 p) -> float:
    float total = 0.0
    float weight = 0.0
    float amplitude = 1.0
    for i in range(octaves):
        total += amplitude * value_noise(p)
        weight += amplitude
        amplitude *= 0.5
        p = p * 2.0 + 0.5
    return total / weight"""

VORONOI_LOOP_GIVEN = """\
            # TODO: keep the second nearest distance in f2 and the nearest point's cell in owner
            f1 = min(f1, d)"""

SOLUTION_VORONOI = """\
            if d < f1:
                f2 = f1
                f1 = d
                owner = cell
            elif d < f2:
                f2 = d"""

VORONOI = """\
%%gmacs "Base" -> "Voronoi"
uniform float scale: hint_range(16, 256) = 96.0
uniform float jitter: hint_range(0, 1) = 1.0
uniform int show: hint_enum("F1", "F2", "F2 - F1", "Cell") = 0

const int F1 = 0
const int F2 = 1
const int F2_MINUS_F1 = 2
const int CELL = 3

{hash}


def pixel(ivec2 at) -> vec4:
    vec2 p = (vec2(at) + 0.5) / scale
    ivec2 home = ivec2(floor(p))
    float f1 = 8.0
    float f2 = 8.0
    ivec2 owner = home
    for dy in range_inclusive(-1, 1):
        for dx in range_inclusive(-1, 1):
            ivec2 cell = home + ivec2(dx, dy)
            vec2 point = vec2(cell) + 0.5 + (random2(cell) - 0.5) * jitter
            float d = length(p - point)
{loop}
    match show:
        case F1:
            return vec4(vec3(f1), 1.0)
        case F2:
            return vec4(vec3(f2), 1.0)
        case F2_MINUS_F1:
            return vec4(vec3(f2 - f1), 1.0)
        case CELL:
            # a third number from the hash of a cell far away
            return vec4(random2(owner), random1(owner + ivec2(-31, 17)), 1.0)"""

SOLUTION_COLOUR = """\
def terrain_colour(float h) -> vec3:
    float t = h - sea
    vec3 lab = STOP_LAB[0]
    for i in range(1, 8):
        float s = clamp((t - STOP_AT[i - 1]) / (STOP_AT[i] - STOP_AT[i - 1]), 0.0, 1.0)
        lab = mix(lab, STOP_LAB[i], s)
    return lab"""

SOLUTION_LIGHT = """\
def light(ivec2 at) -> float:
    float dx = (max(height(at + ivec2(1, 0)), sea) - max(height(at - ivec2(1, 0)), sea)) * 0.5
    float dy = (max(height(at + ivec2(0, 1)), sea) - max(height(at - ivec2(0, 1)), sea)) * 0.5
    vec3 n = normalize(vec3(-relief * dx, -relief * dy, 1.0))
    vec3 l = normalize(vec3(cos(radians(sun)), -sin(radians(sun)), 1.0))
    return 0.25 + 0.75 * max(dot(n, l), 0.0)"""

COLOUR_GIVEN = """\
def terrain_colour(float h) -> vec3:
    # TODO: the OKLab colour of height h between the stops, which are heights above sea
    return STOP_LAB[0]"""

LIGHT_GIVEN = """\
def light(ivec2 at) -> float:
    # TODO: the normal from the heights around at, the water flat, then Lambert
    return 1.0"""

TERRAIN = """\
%%gmacs "Base" -> "{into}"
import linear_srgb_color_space
import oklab_color_space

uniform float scale: hint_range(64, 1024) = 400.0
uniform int octaves: hint_range(1, 8) = 6
uniform float sea: hint_range(0, 1) = 0.5
uniform float relief: hint_range(0, 1000) = 400.0
uniform float sun: hint_range(0, 360) = 135.0

{stops}

{hash}


{value}


{fbm}


def height(ivec2 at) -> float:
    return fbm((vec2(at) + 0.5) / scale)


{colour}


{light}


{pixel}"""

TERRAIN_PIXEL = """\
def pixel(ivec2 at) -> vec4:
    vec3 rgb = oklab_to_linear_srgb(terrain_colour(height(at))) * light(at)
    return vec4(linear_srgb_to_srgb(rgb), 1.0)"""

# The goal can also show the bare fBm the terrain is coloured from.
GOAL_PIXEL = """\
def pixel(ivec2 at) -> vec4:
    if noise_only:
        return vec4(vec3(height(at)), 1.0)
    vec3 rgb = oklab_to_linear_srgb(terrain_colour(height(at))) * light(at)
    return vec4(linear_srgb_to_srgb(rgb), 1.0)"""

SUN = "uniform float sun: hint_range(0, 360) = 135.0"
GOAL = lesson_writer.used_only(
    TERRAIN.replace(SUN, SUN + "\nuniform bool noise_only = false").format(
        into="Goal", stops=STOPS_GMACS, hash=HASH_GMACS, value=VALUE_GMACS, fbm=FBM_GMACS,
        colour=SOLUTION_COLOUR, light=SOLUTION_LIGHT, pixel=GOAL_PIXEL),
    keep=("pixel",))
assert "noise_only = false" in GOAL

STOPS_NP = "\n".join(f"    ({at:.3f}, \"{hex_}\"),{' ' * (9 - len(f'{at:.3f}'))}# {name}" for at, hex_, _, name in STOPS)

HELPERS_NP = f'''\
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


def lab_to_linear(lab):
    """OKLab to linear RGB, not yet encoded to sRGB."""
    return ((lab @ M2_INV.T) ** 3) @ M1_INV.T


def hex_to_rgb(text):
    """"#rrggbb" to three numbers from 0 to 1."""
    return [int(text[i:i + 2], 16) / 255 for i in (1, 3, 5)]


def mix(a, b, t):
    """Same as mix in GLSL: (1 - t) * a + t * b."""
    return (1.0 - t) * a + t * b


def cell_hash(cx, cy):
    """The kernel's cell_hash for whole arrays of cells, in uint32 as on the GPU."""
    h = cx.astype(np.uint32) * np.uint32(1597334677) + cy.astype(np.uint32) * np.uint32(3812015801)
    h = (h ^ (h >> 16)) * np.uint32(2146121005)
    h = (h ^ (h >> 15)) * np.uint32(2221713035)
    return h ^ (h >> 16)


def random1(cx, cy):
    """A number from 0 to 1 for each cell, the same as random1 in the kernel."""
    return (cell_hash(cx, cy) >> 8).astype(np.float32) / np.float32(16777216)


STOPS = [                            # height above the sea, colour
{STOPS_NP}
]
STOP_AT = np.array([at for at, _ in STOPS], dtype=np.float32)
STOP_LAB = to_lab([hex_to_rgb(colour) for _, colour in STOPS])

PARAMS = {{
    "scale": (400.0, 64.0, 1024.0),
    "octaves": (6, 1, 8),
    "sea": (0.5, 0.0, 1.0),
    "relief": (400.0, 0.0, 1000.0),
    "sun": (135.0, 0.0, 360.0),
}}'''

SOLUTION_VALUE_NP = '''\
def value_noise(x, y):
    """Value noise at positions x, y in cells, arrays that broadcast together."""
    cx, cy = np.floor(x), np.floor(y)
    fx, fy = x - cx, y - cy
    fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    cx, cy = cx.astype(np.int64), cy.astype(np.int64)
    a, b = random1(cx, cy), random1(cx + 1, cy)
    c, d = random1(cx, cy + 1), random1(cx + 1, cy + 1)
    return mix(mix(a, b, fx), mix(c, d, fx), fy)'''

FBM_NP = '''\
def fbm(x, y, octaves):
    total, weight, amplitude = 0.0, 0.0, 1.0
    for _ in range(octaves):
        total = total + amplitude * value_noise(x, y)
        weight += amplitude
        amplitude *= 0.5
        x, y = x * 2.0 + 0.5, y * 2.0 + 0.5
    return total / weight'''

SOLUTION_APPLY_NP = '''\
def apply(img, scale, octaves, sea, relief, sun):
    out = img.copy()
    h, w = img.shape[:2]
    x = (np.arange(w)[None, :] + 0.5) / scale      # a row of positions
    y = (np.arange(h)[:, None] + 0.5) / scale      # a column of positions
    height = fbm(x, y, octaves)
    lab = np.stack([np.interp(height - sea, STOP_AT, STOP_LAB[:, k]) for k in range(3)], axis=-1)
    gy, gx = np.gradient(np.maximum(height, sea))
    n = np.stack([-relief * gx, -relief * gy, np.ones_like(height)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    s = np.radians(sun)
    l = np.array([np.cos(s), -np.sin(s), 1.0]) / np.sqrt(2.0)
    light = 0.25 + 0.75 * np.maximum(n @ l, 0.0)
    out[..., :3] = linear_to_srgb(lab_to_linear(lab) * light[..., None])
    out[..., 3] = 1.0
    return out'''


def answer(snippet, lang="python"):
    """A solution block's code fence."""
    return f"```{lang}\n{snippet}\n```"


# --- Title --------------------------------------------------------------------
md(r"""
# Lekce 2: Procedurální šum a krajina

Na konci lekce budete mít generátor krajiny: ostrovy s plážemi, lesy, skalami a sněhem, osvětlené sluncem, které jde posouvat. Nevzniká z žádného obrázku, každý pixel se spočítá jen ze své polohy. Cestou napíšete hodnotový šum, fraktální šum a Voroného šum, ze kterých se skládá většina procedurálních textur.

Spusťte buňku. Otevře nový dokument, jehož jediná vrstva je bílá `Base` přes celé plátno.

<details><summary>📖 K čemu je bílá vrstva</summary>

Kernely z ní nic nečtou, Sára podle ní jen pozná, přes jakou plochu je spustit.
</details>
""")

code("""
import sys
sys.path.insert(0, "lib")            # sara.py and its helpers live in lib/
import numpy as np
import sara

doc, _, pixels = sara.init()
base = doc.new_layer("Base", np.ones_like(pixels))   # white, the size of the canvas
for other in doc.layers():                           # the new document's own layers go
    if other["name"] != "Base":
        doc.layer(other["name"]).delete()
""")

md(r"""
<details><summary>📖 Nové funkce v této lekci</summary>

**Kernel v buňce `%%gmacs`**

| Zápis | Co dělá | Sekce |
|---|---|---|
| `normalize(v)` | vektor délky 1 | 4 |

**NumPy**

| Volání | Co dělá | Sekce |
|---|---|---|
| `a.astype(np.uint32)`, `h >> 16`, `h ^ g` | 32bitová celá čísla bez znaménka a bitové operace po prvcích, jako v kernelu | 5 |
| `np.arange(w)[None, :]`, `np.arange(h)[:, None]` | řádek a sloupec čísel | 5 |
| `np.gradient(a)` | rozdíly sousedních prvků ve směru y a x | 5 |
| `np.linalg.norm(n, axis=-1, keepdims=True)` | délka každého vektoru v poli | 5 |
| `np.maximum(a, b)`, `np.radians(x)` | větší ze dvou po prvcích, stupně na radiány | 5 |

</details>
""")

md(r"""
## Cíl lekce
""")

code(GOAL, hidden=True)

# --- Brainstorming --------------------------------------------------------------
md(r"""
## Jak byste to udělali?

Zaškrtněte v cíli `noise_only`. Krajina je obarvený černobílý šum, výšková mapa: světlé místo je vysoko, tmavé nízko. Jak byste takový šum spočítali, když kernel zná jen polohu pixelu?

<details><summary>💭 Nápady</summary>

Dobrý nápad splní tři věci: sousední pixely mají podobnou výšku, kopce jsou velké i malé a každý pixel si výšku spočítá sám, pokaždé stejně.

<details><summary>Bílý šum: náhodná výška pro každý pixel</summary>

Nehodí se. Sousední pixely spolu nesouvisí, takže kopec by musel vzniknout náhodou ze stovek pixelů najednou, a to se nestane. Vyjde jen zrnění bez tvarů. Osvětlení by ho ještě zesílilo, protože sklon mezi sousedy skáče sem a tam. V lekci 1 bílý šum stačil, protože oko jeho zrnění zprůměruje a vidí jen průměrný odstín. Krajina ale potřebuje právě ty tvary. Uvidíte ho v sekci 1 s volbou `Cells` a `scale = 1`.

</details>

<details><summary>Rozmazat bílý šum</summary>

Funguje to, ale kopec široký sto pixelů potřebuje rozmazání přes stovky pixelů a to je pomalé (lekce 3). Stejný výsledek dají náhodná čísla jen v řídké mřížce a plynulý přechod mezi nimi.

</details>

<details><summary>Náhodná čísla jen v některých bodech a mezi nimi přechod</summary>

Přesně tak. To je hodnotový šum, sekce 1.

</details>

<details><summary>Součet sinusovek</summary>

Pár vln vypadá pravidelně a opakuje se. S mnoha vlnami různých délek a směrů to funguje, a stejná myšlenka, vrstvy čím dál jemnější a slabší, je fraktální šum v sekci 2.

</details>

<details><summary>Kopce na náhodných místech</summary>

Dá ostrovy, ale každý pixel by musel projít všechny kopce. Když dostane každá buňka mřížky jeden bod, stačí projít sousední buňky. Tak pracuje Voroného šum, sekce 3.

</details>

<details><summary>Perlinův šum</summary>

Ano, to je gradientní šum. Napíšeme jednodušší hodnotový šum, Perlinův je v části Co jsme vynechali.

</details>

<details><summary>Diamond-square</summary>

Dá kopce všech velikostí, jenže se počítá po krocích přes celou mřížku, takže pixel nezná svou výšku sám. Navíc bývají vidět švy podél mřížky.

</details>

<details><summary>Výšková data skutečné krajiny</summary>

Věrná, ale musí se uložit, jsou jen tak velká, jak velká jsou, a nejdou přiblížit pod svou přesnost. Procedurální krajina je nekonečná a stačí jí pár parametrů.

</details>

<details><summary>Simulace: tektonika, eroze</summary>

Vypadá nejvěrněji, ale je to mnoho kroků přes celou mapu. Hry jí dolaďují krajinu ze šumu, viz eroze v části Co jsme vynechali.

</details>

</details>

A jak z výškové mapy udělat krajinu?

<details><summary>💭 Nápady</summary>

<details><summary>Barva podle výšky</summary>

Gradientní mapa z lekce 1, jen s víc barvami: voda, písek, tráva, les, skála, sníh. Sekce 4.

</details>

<details><summary>Hladina moře</summary>

Co je pod ní, je voda, a ta je rovná. Barvy se proto měří od hladiny a pod ní se počítá s rovnou plochou. Sekce 4.

</details>

<details><summary>Stín podle sklonu</summary>

Svah natočený ke slunci je světlejší. Normálu dá sklon výškové mapy a jas Lambertův zákon. Sekce 4.

</details>

</details>
""")

# --- 1. Value noise -------------------------------------------------------------
md(r"""
## 1. Hodnotový šum

<details><summary>📖 Náhodná čísla dává hash jako v úvodní lekci</summary>

`cell_hash` v kernelu níže promíchá bity celých souřadnic buňky mřížky, `random1` z nich udělá číslo od 0 do 1 a `random2` dvě. Počítá jen s celými čísly, takže NumPy v sekci 5 dostane bit po bitu stejná čísla.
</details>

Volba `Cells` rozdělí plátno na čtverce o straně `scale` pixelů a každému dá číslo z hashe, se `scale = 1` je to bílý šum. **Hodnotový šum** (*value noise*) dá náhodná čísla jen do rohů buněk a mezi nimi plynule přechází:

@img(value_noise.png, 900, Náhodná čísla v rozích buněk: čtverce, lineární přechod a hladký přechod, nahoře v řezu a dole na ploše)

Bod $p$, měřený v buňkách, leží v buňce $c = \lfloor p \rfloor$ na místě $f = p - c$, obě složky od 0 do 1. V rozích buňky jsou čísla $v_{ij} = \mathrm{random1}(c + (i, j))$. Lineární přechod smíchá zvlášť horní a zvlášť dolní dvojici rohů podle $f_x$ a oba výsledky podle $f_y$, bilineární interpolace (*bilinear interpolation*):

$$\mathrm{noise}(p) = \mathrm{mix}\bigl(\mathrm{mix}(v_{00}, v_{10}, f_x),\; \mathrm{mix}(v_{01}, v_{11}, f_x),\; f_y\bigr)$$

Na hranicích buněk ale nechá zlomy. Hladký přechod nejdřív ohne obě složky $f$ křivkou smoothstep, která má na obou koncích nulovou směrnici, a zlomy zmizí:

$$f \leftarrow 3f^2 - 2f^3$$

### 🎯 Úkol 1: hodnotový šum

Napište `value_noise(p, smoothed)` podle vzorců výše, se `smoothed` hladký přechod, jinak lineární.

<details><summary>💡 Nápověda</summary>

1. Buňka je `ivec2(floor(p))`, poloha uvnitř `fract(p)`.
2. Roh $v_{10}$ je `random1(cell + ivec2(1, 0))`.
3. Smoothstep ohne obě složky najednou, `f` je `vec2`.
</details>

<details><summary>🔑 Řešení</summary>

""" + answer(SOLUTION_VALUE) + r"""
</details>
""")

code(f"""
%%gmacs "Base" -> "Value noise"
uniform float scale: hint_range(1, 256) = 64.0
uniform int mode: hint_enum("Cells", "Linear", "Smooth") = 0

const int CELLS = 0
const int LINEAR = 1
const int SMOOTH = 2

{HASH_GMACS}


def value_noise(vec2 p, bool smoothed) -> float:
    # TODO: the random1 of the cell's four corners, mixed by the position inside it
    return random1(ivec2(floor(p)))


def pixel(ivec2 at) -> vec4:
    vec2 p = (vec2(at) + 0.5) / scale
    match mode:
        case CELLS:
            return vec4(vec3(random1(ivec2(floor(p)))), 1.0)
        case LINEAR:
            return vec4(vec3(value_noise(p, false)), 1.0)
        case SMOOTH:
            return vec4(vec3(value_noise(p, true)), 1.0)
""", solve=[(VALUE_STUB, SOLUTION_VALUE)])

md(r"""
S volbou `Linear` jsou vidět zlomy podél mřížky, se `Smooth` je šum měkký jako mraky.
""")

# --- 2. fBm ---------------------------------------------------------------------
md(r"""
## 2. Fraktální šum

Hodnotový šum má útvary jedné velikosti, krajina má hory, kopce i kameny najednou. **Fraktální šum** (*fractional Brownian motion*, fBm) sečte několik vrstev šumu, oktáv. Každá další je `lacunarity`krát jemnější a `gain`krát slabší:

$$\mathrm{fbm}(p) = \frac{\sum_{i=0}^{n-1} g^i \,\mathrm{noise}(l^i p)}{\sum_{i=0}^{n-1} g^i}$$

Dělení součtem vah drží výsledek mezi 0 a 1.

@img(fbm.png, 900, Čtyři oktávy hodnotového šumu, každá dvakrát jemnější a o polovinu slabší, a jejich součet)

### 🎯 Úkol 2: fBm

Napište `fbm(p)` s `octaves` oktávami. Mezi oktávami polohu kromě zjemnění posuňte o půl buňky, `p = p * lacunarity + 0.5`.

<details><summary>📖 Proč posun o půl buňky</summary>

Jinak by se mřížky všech oktáv sešly v levém horním rohu ve stejné buňce se stejným číslem.
</details>

<details><summary>💡 Nápověda</summary>

1. Potřebujete tři proměnné: součet, součet vah a sílu aktuální oktávy. Síla začíná na 1 a po každé oktávě se násobí `gain`.
2. `p` je parametr funkce a uvnitř ho můžete přepsat.
</details>

<details><summary>🔑 Řešení</summary>

""" + answer(SOLUTION_FBM) + r"""
</details>
""")

code(f"""
%%gmacs "Base" -> "fBm"
uniform float scale: hint_range(16, 1024) = 400.0
uniform int octaves: hint_range(1, 8) = 5
uniform float gain: hint_range(0.2, 0.8) = 0.5
uniform float lacunarity: hint_range(1.5, 3.0) = 2.0

{HASH_GMACS}


{VALUE_GMACS}


def fbm(vec2 p) -> float:
    # TODO: octaves layers of value_noise, each lacunarity times finer and gain times weaker
    return value_noise(p)


def pixel(ivec2 at) -> vec4:
    return vec4(vec3(fbm((vec2(at) + 0.5) / scale)), 1.0)
""", solve=[(FBM_STUB, SOLUTION_FBM)])

md(r"""
S jednou oktávou je obraz stejný jako hladký hodnotový šum, se `gain` kolem 0.8 je šum drsný, kolem 0.3 skoro hladký.
""")

# --- 3. Voronoi -----------------------------------------------------------------
md(r"""
## 3. Voroného šum

Voroného šum (*cellular noise*, podle Steva Worleyho také *Worley noise*) dá do každé buňky mřížky $c$ jeden bod, posunutý ze středu buňky o náhodný kus krát `jitter` $j$:

$$q_c = c + 0.5 + \bigl(\mathrm{random2}(c) - 0.5\bigr)\, j$$

Pro každý pixel $p$ změří vzdálenost k nejbližšímu bodu $F_1$ a k druhému nejbližšímu $F_2$. Bod leží ve své buňce, takže skoro vždy stačí projít 3 × 3 buňky kolem pixelu:

$$F_1 = \min_c \lVert p - q_c \rVert, \qquad F_2 = \min_{c \ne c^*} \lVert p - q_c \rVert$$

kde $c^*$ je buňka nejbližšího bodu.

@img(voronoi.png, 900, Voroného šum po krocích: mřížka s body ve středech, body posunuté náhodně, jeden pixel měří vzdálenosti k bodům 3 × 3 buněk, nejbližší F1 a druhý nejbližší F2, a obrazy F1 a F2 - F1 pro všechny pixely)

<details><summary>📖 Co ukazují F1 a F2 - F1</summary>

$F_1$ je u bodu nula a od něj roste, takže body jsou tmavé skvrny. $F_2 - F_1$ je nula tam, kde má pixel k oběma bodům stejně daleko, tedy na hranicích oblastí, a nakreslí síť jako dlažba nebo kůže.
</details>

### 🎯 Úkol 3: F2 a barva buňky

Upravte cyklus tak, aby si kromě `f1` pamatoval i `f2` a buňku $c^*$ nejbližšího bodu v `owner`. Pak začnou fungovat volby `F2`, `F2 - F1` a `Cell`.

<details><summary>💡 Nápověda</summary>

1. Bod blíž než `f1` je nový nejbližší, a ten dosavadní se tím stane druhým nejbližším.
2. Bod, který není blíž než `f1`, ale je blíž než `f2`, je nový druhý nejbližší.
3. S novým nejbližším bodem si zapamatujte i jeho buňku.
</details>

<details><summary>🔑 Řešení</summary>

Místo řádku s TODO a řádku `f1 = min(f1, d)`:

""" + answer(SOLUTION_VORONOI.replace("\n" + " " * 12, "\n").lstrip()) + r"""
</details>
""")

code(VORONOI.format(hash=HASH_GMACS, loop=VORONOI_LOOP_GIVEN), solve=[(VORONOI_LOOP_GIVEN, SOLUTION_VORONOI)])

md(r"""
`F2 - F1` kreslí tmavé čáry na hranicích buněk a `Cell` obarví každou buňku jednou barvou. S `jitter = 0` jsou buňky zase čtverce.

<details><summary>❓ Otázka: proč si buňky při změně scale nechávají barvu?</summary>

Přepněte na `Cell` a táhněte posuvníkem `scale`. Buňky rostou od levého horního rohu, ale každá si nechává svou barvu. Proč?

<details><summary>🔑 Odpověď</summary>

Barva je `random1` buňky mřížky, ke které patří nejbližší bod, tedy funkce celých souřadnic té buňky. `scale` mění jen to, kolik pixelů buňka zabírá. Buňka (3, 2) zůstává buňkou (3, 2) se stejným hashem i se stejně posunutým bodem.

</details>

</details>
""")

# --- 4. Terrain -----------------------------------------------------------------
md(r"""
## 4. Krajina

fBm je výšková mapa a gradientní mapa z lekce 1 z ní udělá krajinu. Barvy $C_i$ jsou zarážky ve výškách $a_i$ nad hladinou moře `sea` a míchají se v OKLabu:

@img(terrain_stops.png, 900, Barvy krajiny podle výšky nad hladinou moře, od hluboké vody po sníh)

Kernel má výšky zarážek v poli `STOP_AT` a jejich barvy v OKLabu v poli `STOP_LAB`. fBm z úkolu 2 je v něm hotové se `gain = 0.5` a `lacunarity = 2`.

### 🎯 Úkol 4: barvy krajiny

Napište `terrain_colour(h)`, barvu výšky `h` v OKLabu. Pro $t = h - \mathit{sea}$ mezi zarážkami $a_{i-1}$ a $a_i$ je

$$C(t) = \mathrm{mix}\Bigl(C_{i-1},\; C_i,\; \frac{t - a_{i-1}}{a_i - a_{i-1}}\Bigr)$$

Pod první zarážkou je barva první, nad poslední barva poslední.

<details><summary>💡 Nápověda</summary>

1. Zlomek ze vzorce spočítejte pro každý úsek a omezte ho `clamp`: pod úsekem dá 0, nad ním 1.
2. Začněte barvou první zarážky a v cyklu přes úseky ji smíchejte s barvou zarážky `i` podle toho zlomku. Proč to dá správnou barvu i pro `t` vysoko nad úsekem?
</details>

<details><summary>🔑 Řešení</summary>

""" + answer(SOLUTION_COLOUR) + r"""
</details>
""")

md(r"""
Reliéf dá krajině světlo podle Lambertova modelu, jak ho znáte z OpenGL. Nové je jen, odkud vzít normálu: nedává ji síť trojúhelníků, ale sklon výškové mapy, tedy rozdíl výšek sousedů:

$$n = \operatorname{normalize}\bigl(-r\,\partial_x h,\; -r\,\partial_y h,\; 1\bigr), \qquad \partial_x h \approx \frac{h(x+1, y) - h(x-1, y)}{2}$$

a $\partial_y h$ stejně se sousedy dole a nahoře. $r$ je převýšení `relief` a $s$ směr slunce `sun`:

$$l = \operatorname{normalize}(\cos s,\; -\sin s,\; 1)$$

<details><summary>📖 Proč převýšení a minus u sinu</summary>

Výška fBm je od 0 do 1 a plátno má stovky pixelů, takže bez převýšení by byla krajina skoro rovná. Slunce svítí 45° nad obzorem ze směru `sun`, ve stupních proti směru hodinových ručiček od osy x. Osa y v obrázku míří dolů, odtud minus u sinu.
</details>

@img(normal.png, 900, Lambertův model v řezu krajinou: normály n, směr ke slunci l a jas 0.25 + 0.75 max(n · l, 0), svah otočený ke slunci je světlejší a hladina je rovná)

### 🎯 Úkol 5: světlo

Napište `light(at)`, jas $0.25 + 0.75 \max(n \cdot l, 0)$ s normálou a sluncem podle vzorců výše. Voda je rovná, výšku pod hladinou proto berte jako `sea`.

<details><summary>💡 Nápověda</summary>

1. Výšku pixelu dává `height(at)`, souseda vpravo `height(at + ivec2(1, 0))`. Rovnou vodu zařídí `max(..., sea)`.
2. Úhel ve stupních převede na radiány `radians(sun)`.
</details>

<details><summary>🔑 Řešení</summary>

""" + answer(SOLUTION_LIGHT) + r"""
</details>
""")

code(TERRAIN.format(into="Terrain", stops=STOPS_GMACS, hash=HASH_GMACS, value=VALUE_GMACS, fbm=FBM_GMACS,
                    colour=COLOUR_GIVEN, light=LIGHT_GIVEN, pixel=TERRAIN_PIXEL),
     solve=[(COLOUR_GIVEN, SOLUTION_COLOUR), (LIGHT_GIVEN, SOLUTION_LIGHT)])

md(r"""
Svahy otočené ke slunci jsou světlejší a moře je rovné.

<details><summary>📖 Proč se jasem násobí lineární RGB</summary>

Barva se násobí jasem před zakódováním do sRGB, protože světlo se sčítá a násobí lineárně.
</details>
""")

# --- 5. NumPy and Krita ---------------------------------------------------------
md(r"""
## 5. Do Krity

Krita pouští filtry v NumPy, takže generátor je níže přepsaný do NumPy. Spusťte obě buňky. První obsahuje převody barev z úvodní lekce, hash a zarážky, druhá šum a `apply`.
""")

code(HELPERS_NP)

md(r"""
<details><summary>📖 Co se v NumPy mění oproti kernelu</summary>

Kernel počítá jeden pixel, NumPy celé pole najednou. Tím se liší hlavně tyhle řádky:

| Kernel | NumPy |
|---|---|
| `vec2 p = (vec2(at) + 0.5) / scale` | řádek `x` tvaru (1, šířka) a sloupec `y` tvaru (výška, 1), NumPy z nich sám rozšíří pole (výška, šířka) (*broadcasting*) |
| `uint` | `np.uint32`, násobení přeteče stejně, takže hash vyjde stejně |
| `random1(cell + ivec2(1, 0))` | `random1(cx + 1, cy)`, pravý horní roh všech pixelů najednou |
| cyklus přes zarážky v `terrain_colour` | `np.interp` po složkách OKLabu, jako v lekci 1 |
| výšky čtyř sousedů, protože vlákno nevidí výsledky ostatních | `np.gradient` z celé výškové mapy |
| `normalize(n)`, `dot(n, l)` | dělení `np.linalg.norm(n, axis=-1, keepdims=True)`, `n @ l` |

</details>
""")

code(SOLUTION_VALUE_NP + "\n\n\n" + FBM_NP + "\n\n\n" + SOLUTION_APPLY_NP)

code("""
sara.live(apply, PARAMS, source="Base", target="Terrain NumPy")
""")

md(r"""
Vrstva `Terrain NumPy` vypadá stejně jako `Goal`.

### 🎯 Úkol 6: generátor v Kritě

V `pga_filter/effect.py` nahraďte všechno pod řádkem `import numpy as np` obsahem obou buněk s kódem výše a nastavte `TITLE = "Terrain"`. V Kritě založte nový dokument (**File > New**) a spusťte na jeho vrstvě filtr.

<details><summary>❓ Otázka: proč je ve zmenšeném náhledu víc ostrovů?</summary>

Přepněte náhled filtru mezi „Whole region, scaled down“ a „Centre at full size“. Proč je ve zmenšeném náhledu víc menších ostrovů?

<details><summary>🔑 Odpověď</summary>

Zmenšený náhled dostane zmenšené pole, generátor ale měří polohy v pixelech: `scale = 400` je 400 pixelů v jakkoli velkém poli, takže se do menšího vejde víc útvarů. Skutečnou velikost ukáže „Centre at full size“.

</details>

</details>

<details><summary>ℹ️ Krajina v Kritě vyšla světlejší?</summary>

Barvy zarážek jsou v sRGB. Dokument Krity s lineárním profilem (obvykle 32bitová desetinná čísla) je bere jako lineární.
</details>
""")

# --- Summary ------------------------------------------------------------------
md(r"""
## Shrnutí

- Hodnotový šum míchá náhodná čísla z rohů buněk, smoothstep schová zlomy.
- fBm sčítá oktávy a dá útvary všech velikostí.
- Voroného šum měří vzdálenost k rozházeným bodům, $F_2 - F_1$ kreslí hranice oblastí.
- Výšková mapa s gradientní mapou je krajina, normála z jejího sklonu a Lambertův zákon jí dají reliéf.

<details><summary>📖 Co jsme vynechali</summary>

- **Gradientní šum.** Perlinův a simplexový šum dávají do rohů buněk místo čísel směry, takže je v nich mřížka vidět méně.
- **Eroze a vržené stíny.** Obojí potřebuje projít mnoho pixelů nebo kroků simulace, ne jen sousedy.
- **Knihovna `noise`.** Sára má hotové `sa_value_noise`, `sa_value_fbm` a `sa_voronoi` (`import noise`).

</details>

<details><summary>🎯 Bonusové úkoly</summary>

1. **Hřebeny** (*ridged noise*). Sčítejte ve fBm místo šumu $1 - |2\,\mathrm{noise} - 1|$. Ostré hřebeny vypadají jako pohoří.
2. **Deformace prostoru** (*domain warping*). Posuňte polohu o jiný šum, třeba `fbm(p + 4.0 * vec2(fbm(p), fbm(p + 5.2)))`. Vzniknou tvary jako mramor nebo kouř. Inigo Quilez je popisuje v článku [Domain Warping](https://iquilezles.org/articles/warp/) (anglicky).
3. **Dlažba** z Voroného šumu: buňka v barvě z hashe a spáry tam, kde je $F_2 - F_1$ pod prahem.

</details>

<details><summary>📖 Nápady na semestrální práci</summary>

- Generátor textur s rozbalovacím seznamem v `PARAMS` jako `["Clouds", "Marble", "Cells", "Stone"]` a výběrem barev.
- Krajina s paletou podnebí jako rozbalovacím seznamem a sněhem nebo skalou podle sklonu svahu, ne jen podle výšky.

</details>
""")


def build():
    lesson_writer.write(cells, OUT, "l2")

if __name__ == "__main__":
    build()
