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
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": _pictures(text).strip("\n").splitlines(keepends=True)})


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
    (0.008, "#e3d39c", "písek", "sand"),
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
    for i in range(0, octaves):
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
    for i in range(0, octaves):
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
    if show == 1:
        return vec4(vec3(f2), 1.0)
    if show == 2:
        return vec4(vec3(f2 - f1), 1.0)
    if show == 3:
        # a third number from the hash of a cell far away
        return vec4(random2(owner), random1(owner + ivec2(-31, 17)), 1.0)
    return vec4(vec3(f1), 1.0)"""

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


def pixel(ivec2 at) -> vec4:
    vec3 rgb = oklab_to_linear_srgb(terrain_colour(height(at))) * light(at)
    return vec4(linear_srgb_to_srgb(rgb), 1.0)"""

GOAL = TERRAIN.format(into="Goal", stops=STOPS_GMACS, hash=HASH_GMACS, value=VALUE_GMACS, fbm=FBM_GMACS,
                      colour=SOLUTION_COLOUR, light=SOLUTION_LIGHT)

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

VALUE_NP_GIVEN = '''\
def value_noise(x, y):
    """Value noise at positions x, y in cells, arrays that broadcast together."""
    # TODO: the four corners' random1 mixed through the smooth curve, as in the kernel
    return random1(np.floor(x).astype(np.int64), np.floor(y).astype(np.int64))'''

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

FBM_PREVIEW = '''\
h, w = pixels.shape[:2]
noise = fbm((np.arange(w)[None, :] + 0.5) / 400, (np.arange(h)[:, None] + 0.5) / 400, 6)
grey = np.dstack([noise, noise, noise, np.ones_like(noise)])
doc.new_layer("fBm NumPy", grey).show()'''

APPLY_NP_GIVEN = '''\
def apply(img, scale, octaves, sea, relief, sun):
    out = img.copy()
    # TODO: heights from fbm, colours from the stops, light from the normals
    return out'''

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

Spusťte buňku. Otevře v Sáře nový dokument v sRGB místo toho otevřeného a přidá bílou vrstvu `Base` přes celé plátno. Kernely z ní nic nečtou, Sára podle ní jen pozná, přes jakou plochu je spustit.
""")

code("""
import sys
sys.path.insert(0, "lib")            # sara.py and its helpers live in lib/
import numpy as np
import sara

doc, layer, pixels = sara.init()
base = doc.new_layer("Base", np.ones_like(pixels))   # white, the size of the canvas
""")

md(r"""
### Nové funkce v této lekci

Funkce z úvodní lekce shrnuje její příloha. Sloupec Sekce říká, kde se funkce objeví poprvé.

**Kernel v buňce `%%gmacs`**

| Zápis | Co dělá | Sekce |
|---|---|---|
| `uint h`, `1597334677u` | celé číslo bez znaménka, 32 bitů. Výsledek, který se nevejde, přeteče a pokračuje od nuly | 1 |
| `h ^ g`, `h >> 16u`, `h & 65535u` | bitové operace: XOR, posun doprava, AND | 1 |
| `ivec2(floor(p))`, `float(h)`, `uint(x)` | převody mezi typy, `ivec2(floor(p))` je buňka mřížky, ve které leží bod `p` | 1 |
| `for i in range(0, octaves):` | cyklus, jehož mez je parametr | 2 |
| `p *= 2.0`, `total += x` | zkrácené přiřazení jako v Pythonu | 2 |
| `for dy in range_inclusive(-1, 1):` | cyklus včetně horní meze: -1, 0, 1 | 3 |
| `elif` | další podmínka | 3 |
| `length(v)`, `normalize(v)`, `dot(a, b)` | délka vektoru, vektor délky 1, skalární součin | 3, 4 |
| `const vec3 A[8] = vec3[8](...)` | pole vektorů jako konstanta | 4 |
| `radians(x)`, `cos(x)`, `sin(x)` | stupně na radiány a goniometrické funkce | 4 |

**NumPy**

| Volání | Co dělá | Sekce |
|---|---|---|
| `a.astype(np.uint32)` | převod na 32bitová celá čísla bez znaménka, násobení pak přetéká jako na GPU | 5 |
| `h >> 16`, `h ^ g` | bitové operace po prvcích | 5 |
| `np.arange(w)[None, :]`, `np.arange(h)[:, None]` | řádek a sloupec čísel. Počítání s oběma najednou dá celé pole (*broadcasting*) | 5 |
| `np.gradient(a)` | rozdíly sousedních prvků ve směru y a x | 5 |
| `np.linalg.norm(n, axis=-1, keepdims=True)` | délka každého vektoru v poli | 5 |
| `n @ l` | skalární součin každého vektoru v `n` s vektorem `l` | 5 |
| `np.maximum(a, b)`, `np.radians(x)` | větší ze dvou po prvcích, stupně na radiány | 5 |
""")

md(r"""
## Cíl lekce

Buňka níže spustí hotový generátor. Jeho kód je schovaný, protože ho během lekce napíšete sami. Vyzkoušejte hladinu moře `sea`, směr slunce `sun` a počet oktáv `octaves`.
""")

code(GOAL, hidden=True)

# --- 1. Hash and value noise ----------------------------------------------------
md(r"""
## 1. Hash a hodnotový šum

Na GPU počítá každý pixel jiné vlákno a žádné `random()` tam není. Kdyby bylo, dalo by každé spuštění jiný obraz a posuvník by šum pokaždé přeházel. Místo náhody se používá **hash**: funkce, která z celých čísel udělá číslo, které vypadá náhodně, ale pro stejný vstup vyjde vždy stejně. Pixel si tak své „náhodné“ číslo spočítá sám ze své polohy.

`cell_hash` v kernelu níže promíchá bity souřadnic buňky násobením a XORem s posunutými bity, až sousední buňky nemají nic společného. Počítá jen s celými čísly, takže GPU i NumPy dostanou bit po bitu stejný výsledek. `random1` z něj udělá číslo od 0 do 1, `random2` dvě.

Volba `Cells` rozdělí plátno na čtverce o straně `scale` pixelů a každému dá číslo z hashe, se `scale = 1` je to bílý šum. **Hodnotový šum** (*value noise*) dá náhodná čísla jen do rohů buněk a mezi nimi plynule přechází:

@img(value_noise.png, 900, Náhodná čísla v rozích buněk: čtverce, lineární přechod a hladký přechod, nahoře v řezu a dole na ploše)

Lineární přechod nechá na hranicích buněk zlomy. Hladký přechod nejdřív ohne polohu uvnitř buňky $f$ křivkou $3f^2 - 2f^3$ (smoothstep), která má na obou koncích nulovou směrnici, a zlomy zmizí.

### 🎯 Úkol 1: hodnotový šum

Napište `value_noise(p, smoothed)`. `p` je poloha v buňkách: celá část určuje buňku, desetinná část polohu uvnitř ní. Smíchejte čísla ze čtyř rohů buňky podle polohy uvnitř, se `smoothed` přes smoothstep. Dokud funkce chybí, ukazují `Linear` i `Smooth` stejné čtverce jako `Cells`.

<details><summary>💡 Nápověda</summary>

1. Buňka je `ivec2(floor(p))`, poloha uvnitř `fract(p)`, obě souřadnice od 0 do 1.
2. Rohy jsou `cell`, `cell + ivec2(1, 0)`, `cell + ivec2(0, 1)` a `cell + ivec2(1, 1)`.
3. Smíchejte zvlášť horní a zvlášť dolní dvojici rohů podle `f.x` a oba výsledky podle `f.y`. Stejně počítá GPU bilineární filtrování textury.
4. Smoothstep ohne obě souřadnice najednou, `f` je `vec2`.
</details>

<details><summary>🔑 Řešení</summary>

""" + answer(SOLUTION_VALUE) + r"""
</details>
""")

code(f"""
%%gmacs "Base" -> "Value noise"
uniform float scale: hint_range(1, 256) = 64.0
uniform int mode: hint_enum("Cells", "Linear", "Smooth") = 0

{HASH_GMACS}


def value_noise(vec2 p, bool smoothed) -> float:
    # TODO: the random1 of the cell's four corners, mixed by the position inside it
    return random1(ivec2(floor(p)))


def pixel(ivec2 at) -> vec4:
    vec2 p = (vec2(at) + 0.5) / scale
    float v = random1(ivec2(floor(p)))
    if mode > 0:
        v = value_noise(p, mode == 2)
    return vec4(vec3(v), 1.0)
""", solve=[(VALUE_STUB, SOLUTION_VALUE)])

md(r"""
### ✅ Kontrola

S volbou `Linear` jsou vidět rovné zlomy a hvězdice podél mřížky, se `Smooth` je šum měkký jako mraky. Posuvník `scale` mění velikost útvarů a obraz se přitom roztahuje od levého horního rohu.

> **❓ Otázka**
> Když buňku spustíte znovu, nebo celý notebook zítra, vyjde stejný obraz. Proč je to u textury žádoucí? A co by se stalo při tahu posuvníkem, kdyby každý pixel bral číslo z generátoru náhodných čísel?

<details><summary>🔑 Odpověď</summary>

Hash je funkce, pro stejnou buňku dá vždy stejné číslo. Textura je proto stejná při každém spuštění, na každém počítači a na GPU stejně jako v NumPy, takže jde zkontrolovat a místo obrázku stačí uložit hodnoty parametrů. Generátor náhodných čísel by při každém pohybu posuvníkem vrátil jiná čísla a šum by se přeházel, místo aby se plynule měnil. Na GPU navíc vlákno nemá odkud vzít „další“ náhodné číslo, protože vlákna běží současně a nesdílejí stav.
</details>
""")

# --- 2. fBm ---------------------------------------------------------------------
md(r"""
## 2. Fraktální šum

Hodnotový šum má útvary jedné velikosti, krajina má hory, kopce i kameny najednou. **Fraktální šum** (*fractional Brownian motion*, fBm) sečte několik vrstev šumu, oktáv. Každá další je `lacunarity`krát jemnější a `gain`krát slabší:

$$\mathrm{fbm}(p) = \frac{\sum_{i=0}^{n-1} g^i \,\mathrm{noise}(l^i p)}{\sum_{i=0}^{n-1} g^i}$$

Dělení součtem vah drží výsledek mezi 0 a 1.

@img(fbm.png, 900, Čtyři oktávy hodnotového šumu, každá dvakrát jemnější a o polovinu slabší, a jejich součet)

### 🎯 Úkol 2: fBm

Napište `fbm(p)` s `octaves` oktávami. Mezi oktávami polohu kromě zjemnění posuňte o půl buňky, `p = p * lacunarity + 0.5`. Jinak by se mřížky všech oktáv sešly v levém horním rohu ve stejné buňce se stejným číslem. Nedokončená funkce vrací jednu oktávu.

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
### ✅ Kontrola

S jednou oktávou je obraz stejný jako hladký hodnotový šum, každá další přidá jemnější detail. Se `gain` kolem 0.8 je šum drsný a zrnitý, kolem 0.3 skoro hladký.
""")

# --- 3. Voronoi -----------------------------------------------------------------
md(r"""
## 3. Voroného šum

Voroného šum (*cellular noise*, podle Steva Worleyho také *Worley noise*) rozhází po plátně body, v každé buňce mřížky jeden, posunutý o `random2` buňky krát `jitter`. Pro každý pixel změří vzdálenost k nejbližšímu bodu $F_1$ a k druhému nejbližšímu $F_2$:

@img(voronoi.png, 900, Body v buňkách mřížky, vzdálenosti F1 a F2 jednoho pixelu a obrazy F1 a F2 - F1)

$F_1$ je u každého bodu nula a od něj roste, takže body jsou tmavé skvrny a nejsvětlejší jsou místa, kde se stýkají tři buňky. $F_2 - F_1$ je nula všude, kde má pixel k oběma bodům stejně daleko, tedy na hranicích buněk, a nakreslí síť jako dlažba, kůže nebo buňky pod mikroskopem. Kernel prochází 3 × 3 buňky kolem pixelu a v každé počítá vzdálenost k jejímu bodu.

### 🎯 Úkol 3: F2 a barva buňky

Upravte cyklus tak, aby si kromě nejbližší vzdálenosti `f1` pamatoval i druhou nejbližší `f2` a buňku nejbližšího bodu `owner`. Pak začnou fungovat volby `F2`, `F2 - F1` a `Cell`. Nedokončený kernel ukazuje pro `F2` a `F2 - F1` bílou a pro `Cell` barevné čtverce.

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
### ✅ Kontrola

`F2 - F1` kreslí tmavé čáry na hranicích buněk a `Cell` obarví každou buňku jednou barvou. S `jitter = 0` leží body ve středech čtverců mřížky a buňky jsou zase čtverce.

> **❓ Otázka**
> Přepněte na `Cell` a táhněte posuvníkem `scale`. Buňky rostou od levého horního rohu, ale každá si nechává svou barvu. Proč?

<details><summary>🔑 Odpověď</summary>

Barva je `random1` buňky mřížky, ke které patří nejbližší bod, tedy funkce celých souřadnic té buňky. `scale` mění jen to, kolik pixelů buňka zabírá. Buňka (3, 2) zůstává buňkou (3, 2) se stejným hashem i se stejně posunutým bodem.
</details>
""")

# --- 4. Terrain -----------------------------------------------------------------
md(r"""
## 4. Krajina

fBm se dá číst jako výšková mapa: světlé místo je vysoko, tmavé nízko. Gradientní mapa z lekce o barevných prostorech z ní udělá krajinu. Barvy jsou zarážky ve výšce nad hladinou moře `sea` a míchají se v OKLabu:

@img(terrain_stops.png, 900, Barvy krajiny podle výšky nad hladinou moře, od hluboké vody po sníh)

Kernel má zarážky v polích `STOP_AT` (výšky) a `STOP_LAB` (barvy převedené do OKLabu). fBm z úkolu 2 je v něm hotové se `gain = 0.5` a `lacunarity = 2`.

### 🎯 Úkol 4: barvy krajiny

Napište `terrain_colour(h)`, barvu výšky `h` v OKLabu. Leží lineárně mezi dvěma zarážkami, mezi kterými je `h - sea`. Pod první zarážkou je barva první, nad poslední barva poslední. Nedokončená funkce barví celé plátno hlubokou vodou.

<details><summary>💡 Nápověda</summary>

1. Je to gradientní mapa s osmi barvami místo tří. Pro každý úsek mezi zarážkami `i - 1` a `i` spočítejte, kde v něm leží `t = h - sea`, jako číslo od 0 do 1.
2. `clamp` to číslo omezí: pod úsekem dá 0, nad ním 1.
3. Začněte barvou první zarážky a v cyklu přes úseky ji smíchejte s barvou zarážky `i` podle toho čísla. Proč to dá správnou barvu i pro `t` vysoko nad úsekem?
</details>

<details><summary>🔑 Řešení</summary>

""" + answer(SOLUTION_COLOUR) + r"""
</details>
""")

md(r"""
Krajina zatím vypadá jako mapa bez tvaru, reliéf jí dá světlo. Lambertův zákon znáte z OpenGL: plocha s normálou $n$ je světlá podle $\max(n \cdot l, 0)$, kde $l$ míří ke slunci. Normálu tu nedává síť trojúhelníků, ale výšková mapa. Sklon ve směru x je rozdíl výšek souseda vpravo a vlevo, ve směru y souseda dole a nahoře:

$$n = \operatorname{normalize}\bigl(-r\,\partial_x h,\; -r\,\partial_y h,\; 1\bigr), \qquad \partial_x h \approx \frac{h(x+1, y) - h(x-1, y)}{2}$$

$r$ je posuvník `relief`, převýšení. Výška fBm je od 0 do 1, plátno má stovky pixelů, a bez převýšení by byla krajina skoro rovná. Slunce svítí 45° nad obzorem ze směru `sun`, ve stupních proti směru hodinových ručiček od osy x, takže 135° je zleva shora:

$$l = \operatorname{normalize}(\cos s,\; -\sin s,\; 1)$$

Minus u sinu je tam proto, že osa y v obrázku míří dolů.

@img(normal.png, 900, Řez krajinou s normálami a sluncem, svah otočený ke slunci je světlejší)

### 🎯 Úkol 5: světlo

Napište `light(at)`, jas pixelu $0.25 + 0.75 \max(n \cdot l, 0)$. Voda je rovná, výšku pod hladinou proto berte jako `sea`. Nedokončená funkce vrací 1, tedy krajinu bez stínů.

<details><summary>💡 Nápověda</summary>

1. Výšku pixelu dává `height(at)`, souseda vpravo `height(at + ivec2(1, 0))`. Rovnou vodu zařídí `max(..., sea)`.
2. Úhel ve stupních převede na radiány `radians(sun)`.
</details>

<details><summary>🔑 Řešení</summary>

""" + answer(SOLUTION_LIGHT) + r"""
</details>
""")

code(TERRAIN.format(into="Terrain", stops=STOPS_GMACS, hash=HASH_GMACS, value=VALUE_GMACS, fbm=FBM_GMACS,
                    colour=COLOUR_GIVEN, light=LIGHT_GIVEN),
     solve=[(COLOUR_GIVEN, SOLUTION_COLOUR), (LIGHT_GIVEN, SOLUTION_LIGHT)])

md(r"""
### ✅ Kontrola

Plátno je krajina: modré moře, pláže, zelené nížiny, skály a sníh na vrcholcích. Posuvník `sea` zvedá hladinu a ostrovy pod ní mizí. Svahy otočené ke slunci jsou světlejší a moře je rovné, `sun` obchází se sluncem dokola.

> **❓ Otázka**
> Kernel násobí jasem barvu v lineárním RGB, před zakódováním do sRGB. Proč ne až hodnoty sRGB?

<details><summary>🔑 Odpověď</summary>

Jas říká, kolik světla na plochu dopadá, a světlo se násobí a sčítá v lineárním RGB. Hodnoty sRGB jsou zakódované nelineárně: polovina hodnoty sRGB je jen asi pětina světla, takže stíny by vyšly mnohem tmavší, než odpovídá úhlu slunce.
</details>
""")

# --- 5. NumPy and Krita ---------------------------------------------------------
md(r"""
## 5. Do Krity

Teď generátor napíšete v NumPy. Pomocné funkce obsahují převody barev z lekce o barevných prostorech, `lab_to_linear` (převod z OKLabu do lineárního RGB, ještě bez zakódování do sRGB), hash a zarážky. `cell_hash` počítá v `np.uint32` jako GPU: násobení přeteče a pokračuje od nuly, takže vyjdou stejné bity. Buňku stačí spustit.
""")

code(HELPERS_NP)

md(r"""
### 🎯 Úkol 6: hodnotový šum v NumPy

Napište `value_noise(x, y)` pro celé pole poloh najednou. Pod ní je hotové `fbm`, stejné jako v kernelu, a ukázka, která ho nakreslí do vrstvy. `x` je řádek poloh tvaru (1, šířka) a `y` sloupec tvaru (výška, 1). NumPy z nich při počítání sám rozšíří celé pole (výška, šířka), stejně jako při sčítání pole s číslem. Nedokončená funkce vrací čtverce bez přechodů.

<details><summary>💡 Nápověda</summary>

1. Kód je skoro stejný jako v kernelu. `np.floor` vrací buňku jako desetinné číslo, `random1` chce celá čísla: `.astype(np.int64)`.
2. `random1(cx + 1, cy)` spočítá pravý horní roh pro všechny pixely najednou.
3. `mix` z pomocných funkcí funguje i s poli.
</details>

<details><summary>🔑 Řešení</summary>

""" + answer(SOLUTION_VALUE_NP) + r"""
</details>
""")

code(VALUE_NP_GIVEN + "\n\n\n" + FBM_NP + "\n\n\n" + FBM_PREVIEW, solve=[(VALUE_NP_GIVEN, SOLUTION_VALUE_NP)])

md(r"""
### 🎯 Úkol 7: celý generátor

Napište `apply`: výšky z `fbm`, barvy ze zarážek a světlo z normál, stejně jako v kernelu. Obsah `img` generátor nepotřebuje, vezme z něj jen rozměry. Nedokončená funkce vrací vrstvu `Base` beze změny.

Kernel počítal výšku pěti pixelů, aby dostal sklon, protože vlákno nevidí výsledky sousedních vláken. V NumPy máte celou výškovou mapu v jednom poli a sklon spočítá `np.gradient`.

<details><summary>💡 Nápověda</summary>

1. Polohy jsou `x = (np.arange(w)[None, :] + 0.5) / scale` a `y` stejně se sloupcem.
2. Barvy dá `np.interp` po složkách jako v lekci o barevných prostorech, s `STOP_AT` a `STOP_LAB`.
3. `np.gradient(výšky)` vrátí sklon ve směru y a ve směru x, v tomto pořadí. Vodu zarovná `np.maximum`.
4. Normály poskládejte funkcí `np.stack([...], axis=-1)` a vydělte jejich délkou. `n @ l` je skalární součin každé normály s vektorem `l`.
5. Jasem se násobí lineární barva, `lab_to_linear(lab) * jas[..., None]`, a teprve výsledek se zakóduje do sRGB.
</details>

<details><summary>🔑 Řešení</summary>

""" + answer(SOLUTION_APPLY_NP) + r"""
</details>
""")

code(APPLY_NP_GIVEN + """


sara.live(apply, PARAMS, source="Base", target="Terrain NumPy")
""", solve=[(APPLY_NP_GIVEN, SOLUTION_APPLY_NP)])

md(r"""
### ✅ Kontrola

Porovnání s kernelem „Goal“ z úvodu. Kontrola počítá s jeho výchozími hodnotami. Pokud jste s nimi hýbali, spusťte buňku „Goal“ znovu.

`np.gradient` počítá na okrajích plátna sklon jen z jedné strany, kernel z obou, takže krajní pixely se mohou lišit víc. Ostatní by se měly lišit jen zaokrouhlením.
""")

code("""
defaults = {"scale": 400.0, "octaves": 6, "sea": 0.5, "relief": 400.0, "sun": 135.0}
sara.check("Goal", apply(base.read(), **defaults))
""")

md(r"""
### Rychlost
""")

code("""
import time

img = base.read()
times = []
for _ in range(3):
    start = time.perf_counter()
    apply(img, **defaults)
    times.append(time.perf_counter() - start)

t = sara.bench("Goal", runs=20)
print(f"NumPy:      {min(times) * 1000:8.1f} ms")
print(f"GPU kernel: {t.gpu_ms:8.2f} ms")
""")

md(r"""
### 🎯 Úkol 8: generátor v Kritě

Vložte generátor do `pga_filter/effect.py` s `TITLE = "Terrain"`. V Kritě založte nový dokument (**File > New**) a spusťte na jeho vrstvě filtr.

<details><summary>💡 Nápověda</summary>

1. Do `effect.py` patří buňka s pomocnými funkcemi a `PARAMS`, vaše `value_noise`, `fbm` a `apply`. Řádek `import numpy as np` nechte nahoře.
2. Bez výběru zapíše filtr celý dokument. S výběrem začínají souřadnice v rohu výběru, takže výběr dostane stejnou krajinu, jakou má dokument v levém horním rohu.
</details>

> **❓ Otázka**
> Přepněte náhled filtru mezi „Whole region, scaled down“ a „Centre at full size“. Proč je ve zmenšeném náhledu víc menších ostrovů?

<details><summary>🔑 Odpověď</summary>

Pro zmenšený náhled předá šablona funkci `apply` zmenšené pole. Generátor ale počítá polohy v pixelech: `scale = 400` znamená 400 pixelů, ať je pole jakkoli velké, a do menšího pole se tak vejde víc útvarů. Skutečnou velikost ukáže náhled „Centre at full size“, jen v něm uvidíte levý horní roh krajiny, protože i výřez začíná souřadnicí 0. Filtr, který jen mění barvy pixelů, tenhle rozdíl nemá.
</details>

> **ℹ️ Poznámka**
> Barvy zarážek jsou v sRGB. V dokumentu Krity s lineárním profilem, obvykle u 32bitových desetinných čísel na kanál, vyjde krajina světlejší, protože Krita čísla bere jako lineární.
""")

# --- Summary ------------------------------------------------------------------
md(r"""
## Shrnutí

- Hash z celých souřadnic dá číslo, které vypadá náhodně, ale je vždy stejné. Procedurální textura je proto stejná při každém spuštění a na GPU stejná jako v NumPy.
- Hodnotový šum míchá náhodná čísla z rohů buněk, smoothstep schová zlomy na hranicích buněk.
- fBm sčítá oktávy, každou jemnější a slabší, a dá útvary všech velikostí najednou.
- Voroného šum měří vzdálenosti k rozházeným bodům: $F_1$ dá tmavé tečky v bodech, $F_2 - F_1$ hranice buněk a hash buňky její barvu.
- Výšková mapa s gradientní mapou je krajina. Normála z rozdílů sousedních výšek a Lambertův zákon jí dají reliéf.
- Kernel kvůli normále počítá výšku pěti pixelů, NumPy má celou mapu v poli a sklon mu dá `np.gradient`.

### Co jsme vynechali

- **Gradientní šum.** Perlinův a simplexový šum dávají do rohů buněk místo čísel směry a mřížka je v nich vidět méně. Hodnotový šum je jednodušší a pro krajinu stačí.
- **Sousedství 5 × 5.** Mezi 3 × 3 buňkami se výjimečně nenajde bližší bod o dvě buňky dál, hlavně u $F_2$. Přesné je sousedství 5 × 5 s 25 body místo 9.
- **Eroze a vržené stíny.** Údolí vymílá voda, což je simulace o mnoha krocích. Vržený stín potřebuje pro každý pixel projít výšky směrem ke slunci.
- **Knihovna `noise`.** Sára má hotové `sa_value_noise`, `sa_value_fbm` a `sa_voronoi` (`import noise`). Stojí na stejných myšlenkách, jen s hashem z desetinných čísel.

### Bonusové úkoly

1. **Hřebeny** (*ridged noise*). Sčítejte ve fBm místo šumu $1 - |2\,\mathrm{noise} - 1|$. Ostré hřebeny vypadají jako pohoří.
2. **Deformace prostoru** (*domain warping*). Posuňte polohu o jiný šum, třeba `fbm(p + 4.0 * vec2(fbm(p), fbm(p + 5.2)))`. Vzniknou tvary jako mramor nebo kouř. Inigo Quilez je popisuje v článku [Domain Warping](https://iquilezles.org/articles/warp/) (anglicky).
3. **Dlažba** z Voroného šumu: buňka v barvě z hashe a spáry tam, kde je $F_2 - F_1$ pod prahem.
""")

md(r"""
### Nápady na semestrální práci

- Generátor textur s rozbalovacím seznamem v `PARAMS` jako `["Clouds", "Marble", "Cells", "Stone"]` a výběrem barev.
- Krajina s paletou podnebí jako rozbalovacím seznamem a sněhem nebo skalou podle sklonu svahu, ne jen podle výšky.
""")


def build():
    lesson_writer.write(cells, OUT, "l2")

if __name__ == "__main__":
    build()
