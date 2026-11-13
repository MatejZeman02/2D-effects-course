"""Builds the lesson 3 notebook in the lecture folder above this one.

The notebook is generated: edit this file and run it again, never the .ipynb
by hand, or the next build overwrites the edit. The pictures and the test photo
coffee.png come from lesson_03_diagrams.py, which enlarges scikit-image's
coffee photo (CC0) twice, as imgs/3/SOURCES.md says.

Markdown cells are raw strings, so LaTeX keeps single backslashes and braces.
A picture from imgs/3 is written as @img(file, width, alt text).

The solutions are constants near the top, so the markdown shows exactly the
code that test_lesson_03.py runs.

The lesson opens on the problem: the mean of the whole photo, then of a grid
of squares, then of a square around each pixel. Separability is explained by
the associativity of convolution, and the bilateral filter follows the
Gaussian. Shared memory and the second derivative are optional sections at
the end, after the Krita part, so the class path fits 90 minutes.
"""
import re
import textwrap
from pathlib import Path

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
# A square of the grid at the right or bottom edge reaches up to 2 * 32 pixels past it.
MEAN = """\
%%gmacs "coffee" -> "Mean" halo=64
uniform int radius: hint_range(0, 32) = 16
uniform int window: hint_enum("Blocks", "Around the pixel") = 0

const int BLOCKS = 0
const int AROUND = 1


def pixel(ivec2 at) -> vec4:
    int side = 2 * radius + 1
    ivec2 corner                                 # the top left corner of the square
    match window:
        case BLOCKS:
            corner = (at / side) * side          # the square of the grid the pixel lies in
        case AROUND:
            corner = at - ivec2(radius)          # the square centred on the pixel
    vec4 total = vec4(0.0)
    for dy in range(side):
        for dx in range(side):
            total += src[corner + ivec2(dx, dy)]
    return total / float(side * side)"""

CONV_STUB = """\
    # TODO: the 3 x 3 neighbours of at, each times its weight, summed
    return src[at]"""

SOLUTION_CONV = """\
    vec3 total = vec3(0.0)
    for j in range(3):
        for i in range(3):
            total += weight(i, j) * src[at + ivec2(i - 1, j - 1)].rgb
    return vec4(total, src[at].a)"""

CONV = f"""\
%%gmacs "coffee" -> "Convolution"
uniform float weights[9] = {{
     0.0, -1.0,  0.0,
    -1.0,  5.0, -1.0,
     0.0, -1.0,  0.0}}
uniform int symmetry: hint_enum("Off", "Mirror X", "Mirror Y", "Mirror X and Y", "Radial") = 0
uniform bool normalise = true

const int OFF = 0
const int MIRROR_X = 1
const int MIRROR_Y = 2
const int MIRROR_XY = 3
const int RADIAL = 4


def cell(int i, int j) -> float:
    \"\"\"The grid's weight in column i and row j, copied from the top left when symmetry asks.\"\"\"
    int a = min(i, 2 - i)                        # 0 in the left or right column, 1 in the middle one
    int b = min(j, 2 - j)                        # 0 in the top or bottom row, 1 in the middle one
    match symmetry:
        case OFF:
            return weights[j * 3 + i]
        case MIRROR_X:
            return weights[j * 3 + a]            # the right column repeats the left one
        case MIRROR_Y:
            return weights[b * 3 + i]            # the bottom row repeats the top one
        case MIRROR_XY:
            return weights[b * 3 + a]            # every corner repeats the top left 2 x 2
        case RADIAL:
            return weights[min(a, b) * 3 + max(a, b)]  # the corner, the top edge or the centre


def weight(int i, int j) -> float:
    \"\"\"cell(i, j), divided by the sum of all nine when normalise is on and they do not sum to 0.\"\"\"
    float total = 0.0
    for n in range(9):
        total += cell(n % 3, n / 3)
    if normalise and abs(total) > 0.001:
        return cell(i, j) / total
    return cell(i, j)


def pixel(ivec2 at) -> vec4:
{CONV_STUB}"""

# Section 2: the same mean blurs or sharpens in the space the drop-down picks.
SPACES = """\
%%gmacs "coffee" -> "Spaces" halo=8
import linear_srgb_color_space
import oklab_color_space
uniform int effect: hint_enum("Blur", "Sharpen") = 1
uniform int space: hint_enum("sRGB", "Linear RGB", "OKLab") = 0
uniform int radius: hint_range(1, 8) = 3
uniform float amount: hint_range(0, 4) = 2.0

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
    \"\"\"A colour of that space in sRGB again, negative light cut to zero.\"\"\"
    match space:
        case SRGB:
            return c
        case LINEAR:
            return linear_srgb_to_srgb(max(c, vec3(0.0)))
        case OKLAB:
            return linear_srgb_to_srgb(max(oklab_to_linear_srgb(c), vec3(0.0)))


def pixel(ivec2 at) -> vec4:
    int side = 2 * radius + 1
    vec3 mean = vec3(0.0)
    for dy in range(side):
        for dx in range(side):
            mean += into(src[at + ivec2(dx - radius, dy - radius)].rgb)
    mean /= float(side * side)
    vec3 c = into(src[at].rgb)
    match effect:
        case BLUR:
            return vec4(back(mean), 1.0)
        case SHARPEN:
            return vec4(back(c + amount * (c - mean)), 1.0)"""

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
            k *= exp(-dot(d, d) / (2.0 * sigma_color * sigma_color))"""

BILATERAL = f"""\
%%gmacs "coffee" -> "Bilateral" halo=16
import linear_srgb_color_space
import oklab_color_space

uniform float sigma: hint_range(0.5, 4) = 3.0
uniform float sigma_color: hint_range(0.01, 0.5) = 0.05


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

FETCH_STUB = """\
        # TODO: CLAMP, MIRROR and WRAP, each reads src at a position inside the canvas"""

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
{FETCH_STUB}
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

SOBEL_STUB = """\
    # TODO: Sobel's two sums of light over the 3 x 3 neighbours, g.x across and g.y down
    vec2 g = vec2(0.0)"""

SOLUTION_SOBEL = """\
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
{SOBEL_STUB}
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


def rgba(a):
    """An opaque picture from an (h, w, 3) colour array."""
    a = np.asarray(a, dtype=np.float32)
    return np.dstack([a, np.ones(a.shape[:2], np.float32)])


def gaussian_1d(sigma):
    """The weights of a Gaussian of sigma pixels out to 3 sigma, summing to one."""
    r = int(np.ceil(3 * sigma))
    x = np.arange(-r, r + 1)
    k = np.exp(-x ** 2 / (2 * sigma ** 2))
    return k / k.sum()'''

BLUR_NP_GIVEN = '''\
def blur(a, sigma):
    """a, (h, w, c), blurred by a Gaussian of sigma pixels: rows, then columns, edges clamped."""
    # TODO: pad by the radius, then the weighted sum of shifted views, once along each axis
    return a.copy()'''

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
    "amount": (1.0, 0.0, 4.0),
}'''

APPLY_GIVEN = '''\
def apply(img, mode, sigma, amount):
    out = img.copy()
    # TODO: Blur and Sharpen in linear light
    return out'''

SOLUTION_APPLY = '''\
def apply(img, mode, sigma, amount):
    out = img.copy()
    lin = srgb_to_linear(img[..., :3])
    if mode == "Blur":
        lin = blur(lin, sigma)
    else:
        lab = linear_to_oklab(lin)
        lab = lab + amount * (lab - blur(lab, sigma))
        lin = np.maximum(oklab_to_linear(lab), 0.0)
    out[..., :3] = linear_to_srgb(lin)
    return out'''

GOAL = ("# The filter of section 7, hidden because you write it yourself.\n"
        "def _goal():\n"
        + textwrap.indent(lesson_writer.used_only("\n\n".join(piece.replace("\n\n\n", "\n\n") for piece in
                                        [HELPERS_NP, SOLUTION_BLUR_NP, SOLUTION_APPLY]), keep=("apply",)), "    ")
        + "\n\n    return apply\n\n\n"
        + PARAMS_NP
        + '\n\nsara.live(_goal(), PARAMS, source="coffee", target="Goal")')


SHARPEN = [0, -1, 0, -1, 5, -1, 0, -1, 0]


def answer(snippet, lang="python"):
    """A solution block's code fence."""
    return f"```{lang}\n{snippet}\n```"




# --- Title --------------------------------------------------------------------
md(r"""
# Lekce 3: Konvoluce, rozmazání a hrany

Fotku rozmažete, zaostříte a najdete v ní hrany, rychle na GPU i jako filtr v Kritě.
""")

code(f"""
import sys
sys.path.insert(0, "lib")            # sara.py and its helpers live in lib/
import time
import numpy as np
import sara

doc, layer, pixels = sara.init(width={WIDTH}, height={HEIGHT})  # a canvas the size of the photo
photo = doc.import_image("imgs/3/coffee.png")            # a layer called "coffee"
""")

md(r"""
### Nové funkce v této lekci

**Kernel v buňce `%%gmacs`**

| Zápis | Co dělá | Sekce |
|---|---|---|
| `halo=64` na prvním řádku buňky | kernel smí číst až 64 pixelů za okrajem vrstvy | 1 |
| `at / side` | celočíselné dělení, u `ivec2` po složkách | 1 |
| `uniform float weights[9]` | matice 3 × 3 jako mřížka políček pod buňkou, čtená po řádcích | 2 |
| `min`, `max`, `clamp` na `ivec2`, `floor` na `vec2` a převod `ivec2(...)` | po složkách | 5 |
| `kernel`, `image`, `params:`, `dst[pixel] = v` | celý kernel místo funkce `pixel`, s obrázky a parametry | 8 |
| `shared`, `barrier()`, `local_id`, `group_id` | sdílená paměť skupiny vláken, čekání na celou skupinu a čísla vlákna a skupiny | 8 |

**NumPy**

| Volání | Co dělá | Sekce |
|---|---|---|
| `img.mean(axis=(0, 1))` | průměr přes všechny řádky a sloupce, jedna barva | 1 |
| `np.pad(a, šířky, mode=...)` | pole rozšířené o okraje podle pravidla, `"edge"`, `"symmetric"`, `"wrap"` nebo `"constant"` | 7 |
| `a[:, i:i + w]` | výřez pole, pohled na stejná data bez kopie | 7 |
""")

md(r"""
## Cíl lekce
""")

code(GOAL, hidden=True)

# --- 1. The mean ------------------------------------------------------------------
md(r"""
## 1. Rozmazání

Rozmazaná fotka vypadá jako z nezaostřeného objektivu: světlo každého bodu se rozlije do okolí. Pixel má tedy dostat barvu namíchanou z okolních pixelů. Nabízí se průměr. Průměr celé fotky:
""")

code("""
img = photo.read()
whole = np.zeros_like(img) + img.mean(axis=(0, 1))    # every pixel gets the mean of all of them
doc.new_layer("Mean of all", whole).show()
""")

md(r"""
Z fotky zbyla jedna barva. Průměr musí být z menšího okolí. Kernel níže rozdělí fotku na čtverce o straně $2r + 1$ a každý vyplní jeho průměrem. Zmenšujte `radius`:
""")

code(MEAN)

md(r"""
Vznikla mozaika (*pixelation*), ne rozmazání. Dva sousední pixely ze dvou čtverců dostanou úplně jiný průměr, a hranice čtverců jsou proto ostré.

Přepněte `window` na `Around the pixel`. Teď má každý pixel vlastní čtverec se sebou uprostřed. Sousední pixely mají skoro stejné okolí, a výsledek se proto mění plynule. To je **průměr okolí** (*box blur*), `radius` říká, jak moc rozmazává, a s nulou fotku nezmění.

Čtverec u okraje sahá za okraj vrstvy. Kolik pixelů za okrajem kernel smí číst, říká Sáře `halo=64` na prvním řádku buňky. Bez něj Sára čte jen 16 pixelů za okraj a dál dostane průhlednou černou.
""")

# --- 2. Convolution ---------------------------------------------------------------
md(r"""
## 2. Konvoluce

Průměr dává všem sousedům stejnou váhu $1 / (2r + 1)^2$. S různými vahami, i zápornými, umí tentýž výpočet víc než rozmazat. Nová hodnota pixelu je vážený součet jeho okolí a váhy tvoří malou matici, **jádro** (*kernel*, tady ve významu matice vah, ne programu na GPU):

$$g(x, y) = \sum_{j=-r}^{r} \sum_{i=-r}^{r} w(i, j)\, f(x + i, y + j)$$

@img(convolution_window.png, 900, Okno 3 krát 3 nad mřížkou pixelů, každý pixel okna vynásobený svou váhou a součet zapsaný do pixelu uprostřed)

Přísně vzato konvoluce jádro otáčí o 180°, $f(x - i, y - j)$, a vzorec nahoře je korelace (*correlation*). Pro souměrná jádra, jako je rozmazání, vyjdou stejně.

### 🎯 Úkol 1: konvoluce s maticí

Napište konvoluci s libovolnou maticí 3 × 3. `uniform float weights[9]` se pod buňkou ukáže jako mřížka devíti políček, po řádcích: `weights[0]` až `weights[2]` je horní řádek. Váhu políčka ve sloupci `i` a řádku `j` vrací hotová funkce `weight(i, j)`, která zná i dvě volby pod mřížkou. **symmetry** zrcadlí levý horní kout mřížky do ostatních, `Radial` čte jen roh `weights[0]`, hranu `weights[1]` a střed `weights[4]`. **normalise** vydělí váhy jejich součtem, průměr se tak zadá samými jedničkami. Výchozí hodnoty jsou zaostření. Nedokončený kernel vrací původní pixel.

<details><summary>💡 Nápověda</summary>

1. Dva vnořené cykly přes řádek `j` a sloupec `i`, každý od 0 do 2.
2. Políčko `(i, j)` patří sousedovi s posunem `ivec2(i - 1, j - 1)` a jeho váhu vrací `weight(i, j)`.
3. Sčítejte jen barvu, `.rgb`, a alfu vraťte z původního pixelu.
</details>

<details><summary>🔑 Řešení</summary>

""" + answer(SOLUTION_CONV) + r"""
</details>
""")

code(CONV, solve=[(CONV_STUB, SOLUTION_CONV)])

md(r"""
Zkuste do mřížky zadat další jádra:

| Jádro | symmetry | Co vyplnit | Co dělá |
|---|---|---|---|
| průměr | Radial | roh 1, hrana 1, střed 1 | rozmaže |
| zaostření | Radial | roh 0, hrana −1, střed 5 | zvýrazní rozdíl pixelu od sousedů |
| hrany | Radial | roh −1, hrana −1, střed 8 | nechá jen rozdíl od sousedů, plochy zčernají |
| rozmazání do strany | Mirror X and Y | horní řádek 0 0, prostřední 1 1 | rozmaže jen vodorovně |
| reliéf (*emboss*) | Off | −2 −1 0 / −1 1 1 / 0 1 2 | hrany světlé z jedné strany a tmavé z druhé |

@img(kernel_gallery.png, 900, Výřez fotky po konvoluci se čtyřmi jádry z tabulky: průměr, zaostření, hrany a reliéf)

> **❓ Otázka**
> Součet vah průměru i zaostření je 1, součet vah hran je 0. Co součet vah říká o tom, co jádro udělá s jednobarevnou plochou?

<details><summary>🔑 Odpověď</summary>

Na ploše jsou všichni sousedé stejní, takže výsledek je barva plochy krát součet vah. Se součtem 1 plocha zůstane, jak byla, a jádro mění jen místa, kde se sousedé liší. Se součtem 0 plocha zčerná a zbydou jen změny, hrany. Proto `normalise` dělí váhy jejich součtem, a když je součet 0, nechá je být.
</details>

### V jakém prostoru počítat

Konvoluce zatím vážila čísla sRGB. Ta nejsou množství světla, a vážený součet proto nemíchá světlo tak, jak ho míchá rozostřený objektiv. Hotový kernel pod textem rozmaže fotku průměrem okolí a zaostří ji jako pixel plus `amount` krát jeho rozdíl od průměru, a to v prostoru, který vyberete v `space`. Přepínejte prostory a v Sáře se zblízka dívejte na lžičku a na okraj hrnku, u rozmazání s `radius` 8.
""")

code(SPACES)

md(r"""
> **❓ Otázka**
> Ve kterém prostoru vypadá nejlépe rozmazání a ve kterém zaostření? Proč ne v tomtéž?

<details><summary>🔑 Odpověď</summary>

Rozmazání v lineárním RGB. Rozostřený objektiv sčítá světlo, a tak odlesky na lžičce zůstanou velké a jasné, v sRGB a v OKLabu se zmenší a ztmavnou. Zaostření v lineárním RGB naopak nakreslí pod okrajem hrnku černou čáru a nad ním oranžový lem. Rozdíl od průměru je u jasné hrany ve světle obrovský a tmavá strana spadne pod nulu, kde se ořízne. OKLab měří rozdíly tak, jak je vidí oko, takže světlý i tmavý lem vyjdou stejně slabé. sRGB je kódované zhruba podle vnímané světlosti a vyjde skoro stejně. Odteď proto rozmazáváme v lineárním RGB, zaostřujeme a hledáme hrany v OKLabu.
</details>
""")

# --- 3. Gaussian and separability ----------------------------------------------------
md(r"""
## 3. Gauss ve dvou průchodech

Průměr okolí udělá z jasného bodu hranatou skvrnu, protože soused v rohu čtverce váží stejně jako ten vedle. Objektiv rozlije bod do měkké kulaté skvrny. **Gaussovo rozmazání** proto váží souseda podle vzdálenosti:

$$w(i, j) = e^{-\frac{i^2 + j^2}{2\sigma^2}}$$

Ve vzdálenosti $3\sigma$ je váha asi setina té prostřední, kernel proto sčítá jen do poloměru $r = \lceil 3\sigma \rceil$. Kernel počítá v lineárním RGB, jak ukázal konec sekce 2. Poloměr roste se `sigma` až na 24 pixelů, `halo=32` mu dává rezervu.
""")

code(GAUSS_2D)

md(r"""
### Oddělitelnost

Kernel čte $(2r + 1)^2$ sousedů, při $\sigma = 8$ je to 2401 sousedů na pixel. Gaussova váha je ale součin váhy podle $i$ a váhy podle $j$:

$$e^{-\frac{i^2 + j^2}{2\sigma^2}} = e^{-\frac{j^2}{2\sigma^2}} \cdot e^{-\frac{i^2}{2\sigma^2}}$$

Konvoluce řádku $g_\text{ř}$ se sloupcem $g_\text{s}$ dá právě matici těchto součinů, takže Gaussovo jádro je $G = g_\text{ř} * g_\text{s}$. Konvoluce je navíc asociativní. Je to součet součinů a ten jde sčítat v libovolném pořadí (posuny $\mathbf{u}, \mathbf{v}, \mathbf{t}$ jsou dvourozměrné):

$$\bigl((f * a) * b\bigr)(\mathbf{x}) = \sum_{\mathbf{v}} b(\mathbf{v}) \sum_{\mathbf{u}} a(\mathbf{u})\, f(\mathbf{x} - \mathbf{u} - \mathbf{v}) = \sum_{\mathbf{t}} \underbrace{\Bigl(\sum_{\mathbf{u} + \mathbf{v} = \mathbf{t}} a(\mathbf{u})\, b(\mathbf{v})\Bigr)}_{(a * b)(\mathbf{t})} f(\mathbf{x} - \mathbf{t}) = \bigl(f * (a * b)\bigr)(\mathbf{x})$$

Proto

$$f * G = f * (g_\text{ř} * g_\text{s}) = (f * g_\text{ř}) * g_\text{s}$$

Stačí rozmazat každý řádek a výsledek pak rozmazat po sloupcích. Dva průchody čtou $2(2r + 1)$ sousedů místo $(2r + 1)^2$, při $\sigma = 8$ tedy 98 místo 2401. Jádro, které jde takto rozložit na sloupec a řádek, je **oddělitelné** (*separable*).

@img(separable.png, 900, Dvourozměrné Gaussovo jádro jako součin sloupce a řádku a počet čtených sousedů obou postupů v závislosti na poloměru)

> **❓ Otázka**
> Je oddělitelný průměr 3 × 3? A jádro hran −1 −1 −1 / −1 8 −1 / −1 −1 −1?

<details><summary>🔑 Odpověď</summary>

Průměr ano, je to sloupec $(1\;1\;1)^{\mathsf T} / 3$ s řádkem $(1\;1\;1) / 3$. Jádro hran ne. V matici sloupce s řádkem je každý řádek násobkem téhož řádku, ale $(-1\;\,{-1}\;\,{-1})$ a $(-1\;\,8\;\,{-1})$ násobky nejsou.
</details>

### 🎯 Úkol 2: první průchod

Napište první průchod, rozmazání po řádku. Výsledek nechte v lineárním RGB, druhý průchod ho čte tak, jak je, a do sRGB převede až výsledek. V Sáře proto vrstva vypadá tmavší. Nedokončený kernel jen převede pixel do lineárního RGB.

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
Druhý průchod je hotový a čte vrstvu `Blur X`. Po změně `sigma` v prvním průchodu nastavte stejnou hodnotu i ve druhém, sám se nespustí.
""")

code(BLUR_Y)

md(r"""
### Kolik to ušetří

Vrstva `Blur Y` má vypadat stejně jako `Gauss 2D`. Kolik času dva průchody ušetří, změří `sara.bench`. Nechte všude `sigma` na 3.
""")

code("""
two_d = sara.bench("Gauss 2D", runs=10)
rows = sara.bench("Blur X", runs=10)
columns = sara.bench("Blur Y", runs=10)
print(f"one pass, 2D:  {two_d.gpu_ms:8.2f} ms")
print(f"two passes:    {rows.gpu_ms + columns.gpu_ms:8.2f} ms")
""")

md(r"""
> **❓ Otázka**
> Nastavte `sigma` ve všech třech kernelech na 8 a buňku spusťte znovu. Kolikrát se zpomalil `Gauss 2D` a kolikrát dva průchody? Odpovídá to počtu sousedů?

<details><summary>🔑 Odpověď</summary>

Poloměr vzrostl z 9 na 24. Jeden průchod čte $49^2 / 19^2$, asi 6.7krát víc sousedů, dva průchody $98 / 38$, asi 2.6krát víc. Naměřené poměry tomu odpovídají jen zhruba. GPU čte sousedy z rychlé vyrovnávací paměti (*cache*), a u malého poloměru tvoří většinu času spuštění a zápis vrstvy. Dva průchody navíc celou vrstvu `Blur X` jednou zapíšou a znovu přečtou, takže při malém poloměru nemusí vyhrát. Nakonec vraťte `sigma` všude na 3.
</details>
""")

# --- 4. Bilateral -----------------------------------------------------------------
md(r"""
## 4. Rozmazání, které nechá hrany

Gauss vyhladí šum a drobnou texturu, ale stejně rozmaže i hrany, třeba okraj hrnku. Chceme vyhladit plochy a hrany nechat ostré. Soused za hranou má jinou barvu než pixel uprostřed, dáme mu tedy malou váhu. **Bilaterální filtr** (*bilateral filter*) násobí Gaussovu váhu podle vzdálenosti druhou Gaussovou vahou podle rozdílu barev $\mathbf{c}$ v OKLabu:

$$w(i, j) = e^{-\frac{i^2 + j^2}{2\sigma^2}} \cdot e^{-\frac{\lVert \mathbf{c}(x + i,\, y + j) - \mathbf{c}(x, y) \rVert^2}{2\sigma_c^2}}$$

Výsledek se jako u Gausse dělí součtem vah.

@img(bilateral.png, 900, Okolí pixelu na okraji hrnku, váha podle vzdálenosti, váha podle barvy a jejich součin, který za hranou skoro mizí)

### 🎯 Úkol 3: bilaterální filtr

Doplňte barevnou váhu. Funkce `lab(p)` vrátí barvu pixelu `p` v OKLabu, `centre` je barva pixelu uprostřed a `sigma_color` je $\sigma_c$. Nedokončený kernel je Gauss 2D.

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
Porovnejte vrstvy `Bilateral` a `Gauss 2D`. S `sigma_color` 0.05 zůstanou okraj hrnku i lžička ostré, podšálek a dřevo se vyhladí. S velkou `sigma_color` se výsledek blíží `Gauss 2D`.

> **❓ Otázka**
> Proč bilaterální filtr nejde spočítat ve dvou průchodech jako Gauss?

<details><summary>🔑 Odpověď</summary>

Barevná váha závisí na barvě pixelu uprostřed, a každý pixel má proto jiné jádro. Filtr není konvolucí s jedním jádrem, a tak na něj asociativita nejde použít. Dva průchody po řádcích a sloupcích se používají jen jako rychlá náhrada s chybami u šikmých hran.
</details>
""")

# --- 5. Edges of the picture -----------------------------------------------------------
md(r"""
## 5. Za okrajem obrázku

Pixel u okraje nemá sousedy na všech stranách. Co číst za okrajem, je potřeba zvolit:

| Pravidlo | Za okrajem je | V NumPy |
|---|---|---|
| Zero | černá | `np.pad(..., mode="constant")` |
| Clamp | nejbližší pixel na okraji, okraj se natáhne | `mode="edge"` |
| Mirror | obrázek zrcadlený podle okraje | `mode="symmetric"` |
| Wrap | protější okraj, obrázek se opakuje jako dlaždice | `mode="wrap"` |

@img(edge_modes.png, 900, Fotka posunutá doprava dolů a čtyři pravidla pro to, co se objeví za okrajem: černá, natažený okraj, zrcadlo a dlaždice)

Sára sama čte za okrajem nejbližší pixel, tedy Clamp, až do vzdálenosti `halo`. Kernel níže fotku posune o `dx` a `dy` a za okrajem čte podle pravidla `mode`. Rozmazání u okraje čte za okraj stejně, jen méně daleko.

### 🎯 Úkol 4: pravidla okraje

Doplňte do `fetch` větve `CLAMP`, `MIRROR` a `WRAP`. Každá z pozice `p` spočítá pozici uvnitř plátna a přečte ji. Větev `ZERO` je hotová jako příklad. Výchozí pravidlo je Wrap, nedokončený kernel je proto černý.

<details><summary>💡 Nápověda</summary>

1. Clamp: `clamp` funguje i na `ivec2`, po složkách.
2. Mirror pro souřadnici $x$ v plátně šířky $W$: záporné $x$ se zrcadlí na $-1 - x$, $x \ge W$ na $2W - 1 - x$. Obojí jde napsat po složkách funkcemi `max` a `min`.
3. Wrap je zbytek po dělení, který je i pro záporné $x$ od 0 do $W - 1$. Operátor `%` v GLSL na záporných číslech výsledek nezaručuje. Pomůže $x - W \lfloor x / W \rfloor$, s dělením ve `float`.
</details>

<details><summary>🔑 Řešení</summary>

""" + answer(SOLUTION_FETCH.replace("\n        ", "\n").lstrip()) + r"""
</details>
""")

code(EDGES, solve=[(FETCH_STUB, SOLUTION_FETCH)])

md(r"""
> **❓ Otázka**
> Které pravidlo se hodí k rozmazání fotky, které k rozmazání dlaždice textury a proč je Zero u rozmazání skoro vždy špatně?

<details><summary>🔑 Odpověď</summary>

U fotky Mirror nebo Clamp: za okrajem pokračuje něco podobného tomu, co je u okraje. Clamp u velkého poloměru dělá pruhy z natažených pixelů, Mirror ne. Textura, která se má opakovat, potřebuje Wrap, jinak by se na švech dlaždic objevil skok. Zero přimíchá do okrajů černou a okraje rozmazané fotky ztmavnou. Fourierova transformace v lekci o frekvencích bere obrázek vždy jako dlaždici, tedy Wrap.
</details>
""")

# --- 6. Sobel ---------------------------------------------------------------------
md(r"""
## 6. Hrany

Hrana je místo, kde se rychle mění světlost. Změnu měří derivace a na pixelech je to rozdíl sousedů: jádro $(-1\;\,0\;\,1)$ odečte levého souseda od pravého. Jeden řádek ale zachytí i šum. **Sobelův operátor** proto rozdíl zprůměruje přes tři řádky s vahami $(1\;\,2\;\,1)$. Podle sekce 3 je to konvoluce sloupce s řádkem, oddělitelné jádro:

$$G_x = \begin{pmatrix} 1 \\ 2 \\ 1 \end{pmatrix} * \begin{pmatrix} -1 & 0 & 1 \end{pmatrix} = \begin{pmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{pmatrix} \qquad G_y = G_x^{\mathsf T}$$

Dvojice derivací $(g_x, g_y)$ je **gradient**. Jeho délka je síla hrany, jeho směr míří od tmavé strany ke světlé, kolmo na hranu.

@img(sobel.png, 900, Sobelova jádra Gx a Gy, gradient na hraně kolmo na ni a barevné kolo směrů, které kernel používá)

### 🎯 Úkol 5: Sobel

Spočítejte gradient světlosti OKLab L, `light(p)`, v proměnné `g`: `g.x` je součet s jádrem $G_x$, `g.y` s jádrem $G_y$. Funkce `shade` obarví hranu podle směru, když zapnete `direction`. Nedokončený kernel vrací černou.

<details><summary>💡 Nápověda</summary>

1. Dva vnořené cykly přes posuny `i` a `j` od −1 do 1, jako v úkolu 1.
2. Váha $G_x$ v políčku `(i, j)` je `i` krát 2 v prostředním řádku a 1 v ostatních, tedy `i * (2 - abs(j))`. Váha $G_y$ je totéž s prohozenými `i` a `j`.
3. Oba součty jdou najednou: `g += light(...) * vec2(váha x, váha y)`.
</details>

<details><summary>🔑 Řešení</summary>

""" + answer(SOLUTION_SOBEL) + r"""
</details>
""")

code(SOBEL, solve=[(SOBEL_STUB, SOLUTION_SOBEL)])

md(r"""
> **❓ Otázka**
> Zapněte `direction`. Proč mají dvě strany tenké čáry, třeba okraje lžičky, opačné barvy?

<details><summary>🔑 Odpověď</summary>

Barva je směr gradientu, od tmavší strany ke světlejší. Tenká světlá čára má tmavé okolí na obou stranách: na jedné straně světlost roste, na druhé klesá, a gradienty míří proti sobě. Opačné směry dostanou na barevném kole opačné barvy. Cannyho detektor hran to využívá: hranu ztenčí na jeden pixel tak, že nechá jen místa, kde je síla největší ve směru gradientu.
</details>
""")

# --- 7. NumPy and Krita -----------------------------------------------------------
md(r"""
## 7. Do Krity

Krita kernely nespouští a NumPy nemá vlákno na pixel, umí ale celé pole najednou. Posunutý obrázek je **výřez** rozšířeného pole, `p[:, i:i + w]`, pohled na tatáž data bez kopie. Vážený součet $2r + 1$ výřezů je jeden průchod rozmazání.

@img(numpy_shifts.png, 760, Řádek rozšířený o okraje a pět výřezů posunutých o jeden pixel, každý vynásobený svou vahou a sečtený)

Buňka pod textem má pomocné funkce: převody sRGB a OKLabu, `rgba` a váhy Gaussova jádra `gaussian_1d`. Stačí ji spustit.
""")

code(HELPERS_NP)

md(r"""
### 🎯 Úkol 6: rozmazání v NumPy

Napište `blur(a, sigma)`: rozmazání po řádcích a pak po sloupcích, okraj podle pravidla Clamp. Váhy dává `gaussian_1d(sigma)`, pole délky $2r + 1$. Výsledek se zapíše do vrstvy `Blur NumPy`, která má vypadat jako `Gauss 2D` se `sigma` 3.

<details><summary>💡 Nápověda</summary>

1. `np.pad(a, ((0, 0), (r, r), (0, 0)), mode="edge")` rozšíří pole o `r` pixelů vlevo a vpravo.
2. Výřez `p[:, i:i + w]` je obrázek posunutý o `i - r` pixelů. Sečtěte `k[i]` krát výřez pro všechna `i`.
3. Pak totéž pro sloupce, s rozšířením v ose 0.
</details>

<details><summary>🔑 Řešení</summary>

""" + answer(SOLUTION_BLUR_NP) + r"""
</details>
""")

code(BLUR_NP_GIVEN + f"""


lin = srgb_to_linear(photo.read()[..., :3])
start = time.perf_counter()
soft = blur(lin, {SIGMA})
took = time.perf_counter() - start
doc.new_layer("Blur NumPy", rgba(linear_to_srgb(soft)))
print(f"NumPy blur: {{took * 1000:.1f}} ms")
""", solve=[(BLUR_NP_GIVEN, SOLUTION_BLUR_NP)])

md(r"""
**Zaostření** (*unsharp mask*) přičte k obrázku jeho rozdíl od rozmazaného obrázku, $f + a\,(f - \text{blur}(f))$. Rozdíl jsou detaily menší než $\sigma$ a `amount` $a$ říká, kolikrát je zesílit. Proč to funguje, ukazuje nepovinná sekce 9.

### 🎯 Úkol 7: filtr

Napište `apply` s režimy `Blur` v lineárním RGB a `Sharpen` v OKLabu. Zaostřená barva může vyjít mimo sRGB, záporné světlo ořízněte na nulu.

<details><summary>💡 Nápověda</summary>

1. Převeďte barvu do lineárního RGB. `Blur` ji rozmaže funkcí `blur`.
2. `Sharpen` ji převede `linear_to_oklab`, přičte `amount` krát rozdíl od `blur(lab, sigma)` a vrátí `oklab_to_linear`, oříznuté `np.maximum(..., 0.0)`.
3. Výsledek převeďte do sRGB, alfa zůstává z `img`.
</details>

<details><summary>🔑 Řešení</summary>

""" + answer(SOLUTION_APPLY) + r"""
</details>
""")

code(PARAMS_NP + "\n\n\n" + APPLY_GIVEN + """


sara.live(apply, PARAMS, source="coffee", target="Filter NumPy")
""", solve=[(APPLY_GIVEN, SOLUTION_APPLY)])

md(r"""
### 🎯 Úkol 8: filtr v Kritě

Vložte filtr do `pga_filter/effect.py` s `TITLE = "Blur and sharpen"` a vyzkoušejte ho na vlastní fotce. Porovnejte rychlost s Gaussovým rozmazáním, které má Krita vestavěné.

<details><summary>💡 Nápověda</summary>

1. Do `effect.py` patří pomocné funkce, vaše `blur`, `PARAMS` a `apply`.
2. Náhled počítá jen malou kopii, pomalé je až Apply na celé fotce. Zkuste filtr nejdřív na výběru. Ve zmenšeném náhledu „Whole region, scaled down“ vypadá rozmazání silnější, skutečnou sílu ukáže „Centre at full size“.
</details>

> **❓ Otázka**
> Vystřižený předmět na průhledné vrstvě má po rozmazání kolem sebe tmavý lem. Proč, a jak mu předejít?

<details><summary>🔑 Odpověď</summary>

Průhledné pixely mají nějakou barvu, obvykle černou, a rozmazání ji přimíchá k barvě předmětu, i když ji nikdo nevidí. Pomůže rozmazávat barvu vynásobenou alfou (*premultiplied alpha*), rozmazat i alfu a výslednou barvu alfou zase vydělit. Průhledný pixel pak k barvě nepřidá nic.
</details>
""")

# --- 8. Shared memory (optional) ---------------------------------------------------
md(r"""
## 8. Sdílená paměť (nepovinné)

V prvním průchodu čte každé vlákno $2r + 1$ pixelů řádku a sousední vlákna čtou skoro tytéž pixely. Při $\sigma = 8$ přečte GPU každý pixel 49krát a 49krát ho převede ze sRGB. Vlákna jedné **skupiny** (*workgroup*) mají společnou rychlou **sdílenou paměť** (*shared memory*). Skupina si do ní řádek načte jednou, každé vlákno jeden nebo dva pixely, počká na ostatní a pak sčítá ze sdílené paměti.

@img(shared_tile.png, 900, Skupina 64 vláken načte do sdílené paměti svých 64 pixelů a 24 pixelů na každé straně, pak každé vlákno sečte 49 hodnot ze sdílené paměti)

Funkce `pixel` sdílenou paměť nemá, kernel je proto napsaný celý: řádek `kernel` se jménem, obrázky `src` a `dst`, parametry pod `params:` a skupina 64 vláken v řádku. `local_id` je číslo vlákna ve skupině, `group_id` číslo skupiny a `pixel` pozice vlákna v obrázku. `barrier()` počká, až všechna vlákna skupiny dojdou na totéž místo, jinak by vlákno mohlo číst, kam soused ještě nezapsal. Parametr celého kernelu nemá výchozí hodnotu, dává mu ji první řádek buňky, `sigma=3`.

Celý kernel vidí obrázek včetně okraje `halo`, takže jeho `pixel` je posunutý o 32 proti pozici na plátně a `image_size(src)` je o 64 větší než plátno. Zápis do `dst` na stejný `pixel` přistane na správném místě.
""")

code(SHARED)

code("""
shared = sara.bench("Blur X shared", runs=10)
plain = sara.bench("Blur X", runs=10)
print(f"first pass, plain:   {plain.gpu_ms:8.3f} ms")
print(f"first pass, shared:  {shared.gpu_ms:8.3f} ms")
""")

md(r"""
> **❓ Otázka**
> Zkuste měření s `sigma` 1, 3 a 8 v obou kernelech. Kdy se sdílená paměť vyplatí a kdy ne?

<details><summary>🔑 Odpověď</summary>

Vyplatí se, když vlákna čtou hodně stejných pixelů a čtení je drahé: velký poloměr a převod ze sRGB u každého čtení. Kernel se sdílenou pamětí čte a převádí každý pixel nanejvýš dvakrát, ať je poloměr jakýkoli, a jeho čas skoro neroste. Při malém poloměru je rozdíl malý, protože GPU drží nedávno čtené pixely ve vyrovnávací paměti samo, a načtení do sdílené paměti a čekání na `barrier()` něco stojí. Velikost sdíleného pole musí být známá při překladu, a poloměr je proto omezený konstantou `R_MAX`.
</details>
""")

# --- 9. Second derivative (optional) ----------------------------------------------
md(r"""
## 9. Druhá derivace (nepovinné)

Sobel měří první derivaci, strmost. Hrana je tam, kde je strmost největší, a v tom místě má **druhá derivace** nulu a mění znaménko: na tmavé straně hrany je kladná, na světlé záporná.

@img(second_derivative.png, 900, Měkká hrana vzorkovaná po pixelech, její první derivace s vrcholem na hraně, druhá derivace s průchodem nulou na hraně a hrana minus druhá derivace se zákmity na obou stranách)

Jádro pro druhou derivaci dá Taylorův rozvoj. Pro souseda ve vzdálenosti $h$ na obě strany:

$$f(x \pm h) = f(x) \pm h\,f'(x) + \frac{h^2}{2}\,f''(x) \pm \frac{h^3}{6}\,f'''(x) + O(h^4)$$

Po sečtení obou řádků se členy s lichou mocninou $h$ odečtou:

$$f(x + h) + f(x - h) = 2 f(x) + h^2 f''(x) + O(h^4)$$

$$f''(x) = \frac{f(x + h) - 2 f(x) + f(x - h)}{h^2} + O(h^2)$$

S krokem jednoho pixelu, $h = 1$, je to jádro $(1\;\,{-2}\;\,1)$. Ve 2D se druhé derivace v $x$ a v $y$ sečtou do **Laplaceova operátoru** $\nabla^2 f = f_{xx} + f_{yy}$, a sečtou se i jejich jádra:

$$\nabla^2 \approx \begin{pmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{pmatrix}$$

Jádro hran z tabulky v úkolu 1 je záporný Laplace, který bere i sousedy na úhlopříčkách. Zaostření z úkolu 1 je identita minus Laplace:

$$\begin{pmatrix} 0 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 0 \end{pmatrix} - \begin{pmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{pmatrix} = \begin{pmatrix} 0 & -1 & 0 \\ -1 & 5 & -1 \\ 0 & -1 & 0 \end{pmatrix}$$

Odečtení druhé derivace hranu zostří: na tmavé straně je $f''$ kladná a pixel ztmavne, na světlé je záporná a pixel zesvětlí. Na obrázku vpravo je to zákmit na obou stranách hrany.

Druhá derivace ale zesílí nejvíc ty nejjemnější detaily. Vlna $\cos \omega x$ vyjde z jádra $(1\;\,{-2}\;\,1)$ jako $-4\sin^2(\omega / 2)\,\cos\omega x$, tedy vynásobená číslem, které roste s frekvencí. Nejvíc tedy zesílí šum, a proto se Laplace počítá z rozmazaného obrázku (*Laplacian of Gaussian*, LoG).

Zaostření z úkolu 7 (*unsharp mask*) je také identita minus Laplace, jen ve větším měřítku. Taylorův rozvoj rozmazání do druhého řádu, zprůměrovaný s Gaussovými vahami, dá

$$\text{blur}(f) \approx f + \frac{\sigma^2}{2}\,\nabla^2 f, \qquad f + a\,(f - \text{blur}(f)) \approx f - a\,\frac{\sigma^2}{2}\,\nabla^2 f$$

> **❓ Otázka**
> Zadejte do mřížky v úkolu 1 Laplaceovo jádro, 0 1 0 / 1 −4 1 / 0 1 0. Proč je vrstva skoro celá černá, a kde svítí?

<details><summary>🔑 Odpověď</summary>

Součet vah je 0, takže každá plocha zčerná. Svítí jen tmavá strana hran, kde je druhá derivace kladná, a jemná textura. Na světlé straně hran je výsledek záporný a obrazovka ho ukáže jako černou. Nakonec vraťte do mřížky zaostření.
</details>
""")

# --- Summary ------------------------------------------------------------------
md(r"""
## Shrnutí

- Rozmazání míchá pixel s okolím: průměr celé fotky je jedna barva, průměr v mřížce mozaika, průměr okolí každého pixelu rozmazání.
- Konvoluce je vážený součet okolí. Součet vah 1 zachová plochy, součet 0 nechá jen změny.
- Konvoluce je asociativní, a jádro, které je konvolucí sloupce s řádkem, jde proto spočítat ve dvou průchodech: $2(2r + 1)$ sousedů místo $(2r + 1)^2$. Takové je Gaussovo jádro, průměr i Sobel.
- Rozmazání míchá světlo, a proto počítá v lineárním RGB. Zaostření a hrany měří rozdíly, jak je vidí oko, a proto v OKLabu.
- Bilaterální filtr váží souseda i podle rozdílu barev a hrany nerozmaže. Jádro se mění od pixelu k pixelu, a dva průchody proto nejdou.
- Za okrajem je nutné zvolit pravidlo: Zero, Clamp, Mirror nebo Wrap. Fotce se hodí Mirror nebo Clamp, textuře Wrap.
- Sobelův operátor odhadne gradient světlosti. Jeho délka je síla hrany, směr míří kolmo na hranu.
- NumPy počítá konvoluci jako vážený součet posunutých výřezů celého pole.

### Co jsme vynechali

- **Rozmazání s proměnným poloměrem.** Hloubka ostrosti nebo rozmazání podle masky mění $\sigma$ pixel od pixelu a jádro pak není oddělitelné.
- **Rychlé aproximace.** Několik průměrů za sebou se blíží Gaussovi. Kawaseho rozmazání a pyramida zmenšených obrázků jsou rychlé způsoby, jak hry dělají rozmazání pro bloom.
- **Cannyho detektor hran.** Gaussovo rozmazání, Sobel, ztenčení hran na jeden pixel a dvě prahové hodnoty dají dohromady čisté čáry místo šedých pruhů.
- **Konvoluce přes Fourierovu transformaci.** Pro velká jádra je rychlejší násobit spektra, jak ukáže lekce o frekvencích.

### Bonusové úkoly

1. **Průměr v konstantním čase.** Napište průměr okolí v NumPy jako rozdíl kumulativních součtů, `np.cumsum`, takže čas nezávisí na poloměru. Třikrát za sebou dá skoro Gausse.
2. **Bilaterální filtr v NumPy.** Přidejte do filtru pro Kritu režim `Bilateral`. Jako u `blur` je to vážený součet výřezů, jen váha každého výřezu je pole, ne číslo.
3. **Ostření jen na hranách.** Použijte sílu hrany ze Sobela jako masku zaostření, aby se nezesílil šum na plochách.
""")

md(r"""
### Nápady na semestrální práci

- Rozostření podle masky: plugin rozmaže vrstvu podle světlosti druhé vrstvy nebo výběru, jako tilt-shift fotografie.
- Detektor hran s volbou operátoru (Sobel, Scharr, Prewitt) a výstupem jako čárová kresba nebo maska výběru.
- Vyhlazení pleti nebo šumu bilaterálním filtrem s maskou, kde se vyhlazuje.
""")


def build():
    lesson_writer.write(cells, OUT, "l3")


if __name__ == "__main__":
    build()
