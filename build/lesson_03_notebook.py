"""Builds the lesson 3 notebook in the lecture folder above this one.

The notebook is generated: edit this file and run it again, never the .ipynb
by hand, or the next build overwrites the edit. The pictures and the test photo
coffee.png come from lesson_03_diagrams.py, which enlarges scikit-image's
coffee photo (CC0) twice, as imgs/3/SOURCES.md says.

Markdown cells are raw strings, so LaTeX keeps single backslashes and braces.
A picture from imgs/3 is written as @img(file, width, alt text).

The solutions are constants near the top, so the markdown shows exactly the
code that test_lesson_03.py runs.
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
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": _pictures(text).strip("\n").splitlines(keepends=True)})


def code(text, hidden=False, solve=()):
    metadata = {"jupyter": {"source_hidden": True}} if hidden else {}
    cell = {"cell_type": "code", "metadata": metadata, "execution_count": None, "outputs": [],
            "source": lesson_writer.lines(text)}
    if solve:
        cell["solution"] = lesson_writer.solved(text, solve)
    cells.append(cell)


# --- The kernels ------------------------------------------------------------------
BOX = """\
%%gmacs "coffee" -> "Box blur" halo=32
uniform int radius: hint_range(0, 32) = 4


def pixel(ivec2 at) -> vec4:
    vec4 total = vec4(0.0)
    for dy in range_inclusive(-radius, radius):
        for dx in range_inclusive(-radius, radius):
            total += src(at + ivec2(dx, dy))
    int side = 2 * radius + 1
    return total / float(side * side)"""

CONV_STUB = """\
    # TODO: the 3 x 3 neighbours of at, each times its weight, summed
    return src(at)"""

SOLUTION_CONV = """\
    vec3 total = vec3(0.0)
    for j in range(0, 3):
        for i in range(0, 3):
            total += weights[j * 3 + i] * src(at + ivec2(i - 1, j - 1)).rgb
    return vec4(total, src(at).a)"""

CONV = f"""\
%%gmacs "coffee" -> "Convolution"
uniform float weights[9] = {{
     0.0, -1.0,  0.0,
    -1.0,  5.0, -1.0,
     0.0, -1.0,  0.0}}


def pixel(ivec2 at) -> vec4:
{CONV_STUB}"""

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
            total += k * srgb_to_linear_srgb(src(at + ivec2(dx, dy)).rgb)
            weight += k
    return vec4(linear_srgb_to_srgb(total / weight), 1.0)"""

BLUR_X_STUB = """\
    # TODO: the Gaussian sum along the row, divided by the sum of the weights
    return vec4(srgb_to_linear_srgb(src(at).rgb), 1.0)"""

SOLUTION_BLUR_X = """\
    int r = int(ceil(3.0 * sigma))
    vec3 total = vec3(0.0)
    float weight = 0.0
    for dx in range_inclusive(-r, r):
        float k = exp(-float(dx * dx) / (2.0 * sigma * sigma))
        total += k * srgb_to_linear_srgb(src(at + ivec2(dx, 0)).rgb)
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
        total += k * src(at + ivec2(0, dy)).rgb
        weight += k
    return vec4(linear_srgb_to_srgb(total / weight), 1.0)"""

FETCH_STUB = """\
    # TODO: Clamp, Mirror and Wrap, each a position inside the canvas
    return vec4(0.0, 0.0, 0.0, 1.0)"""

SOLUTION_FETCH = """\
    ivec2 q = p
    if mode == 1:
        q = clamp(p, ivec2(0), SIZE - 1)
    elif mode == 2:
        q = min(max(p, -1 - p), 2 * SIZE - 1 - p)
    else:
        q = p - SIZE * ivec2(floor(vec2(p) / vec2(SIZE)))
    return src(q)"""

SHIFT = (160, 96)             # the edge kernel's opening shift, dx and dy

EDGES = f"""\
%%gmacs "coffee" -> "Shifted" halo=256
uniform int mode: hint_enum("Zero", "Clamp", "Mirror", "Wrap") = 3
uniform int dx: hint_range(-256, 256) = {SHIFT[0]}
uniform int dy: hint_range(-256, 256) = {SHIFT[1]}

const ivec2 SIZE = ivec2({WIDTH}, {HEIGHT})


def fetch(ivec2 p) -> vec4:
    \"\"\"The pixel at p, wherever p lies, by the edge rule mode picks.\"\"\"
    if mode == 0:
        if p.x < 0 or p.y < 0 or p.x >= SIZE.x or p.y >= SIZE.y:
            return vec4(0.0, 0.0, 0.0, 1.0)
        return src(p)
{FETCH_STUB}


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
        row[i] = srgb_to_linear_srgb(image_load(src, p).rgb)
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
    image_store(dst, pixel, vec4(total / weight, 1.0))"""

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
    return linear_srgb_to_oklab(srgb_to_linear_srgb(src(p).rgb)).x


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
M2_L = np.array([0.2104542553, 0.7936177850, -0.0040720468], dtype=np.float32)   # OKLab L from cube-rooted LMS


def lightness(img):
    """OKLab L of an (h, w, 4) picture in sRGB, an (h, w) array."""
    return np.cbrt(srgb_to_linear(img[..., :3]) @ M1.T) @ M2_L


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
    return k / k.sum()'''

BLUR_NP_GIVEN = '''\
def blur(a, sigma):
    """a, (h, w, c) in linear light, blurred by a Gaussian of sigma pixels: rows, then columns, edges clamped."""
    # TODO: pad by the radius, then the weighted sum of shifted views, once along each axis
    return a.copy()'''

SOLUTION_BLUR_NP = '''\
def blur(a, sigma):
    """a, (h, w, c) in linear light, blurred by a Gaussian of sigma pixels: rows, then columns, edges clamped."""
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
    soft = blur(lin, sigma)
    if mode == "Blur":
        lin = soft
    else:
        lin = np.maximum(lin + amount * (lin - soft), 0.0)
    out[..., :3] = linear_to_srgb(lin)
    return out'''

GOAL = ("# The filter of section 6, hidden because you write it yourself.\n"
        "def _goal():\n"
        + textwrap.indent("\n\n".join(piece.replace("\n\n\n", "\n\n") for piece in
                                        [HELPERS_NP, SOLUTION_BLUR_NP, SOLUTION_APPLY]), "    ")
        + "\n\n    return apply\n\n\n"
        + PARAMS_NP
        + '\n\nsara.live(_goal(), PARAMS, source="coffee", target="Goal")')

DEFAULTS = '{"mode": "Blur", "sigma": 3.0, "amount": 1.0}'

SHARPEN = [0, -1, 0, -1, 5, -1, 0, -1, 0]


def answer(snippet, lang="python"):
    """A solution block's code fence."""
    return f"```{lang}\n{snippet}\n```"


# --- Title --------------------------------------------------------------------
md(r"""
# Lekce 3: Konvoluce, rozmazání a hrany

Na konci lekce budete mít v Kritě filtr, který fotku rozmaže nebo zaostří, a rychlý, protože rozmazání rozdělíte na dva průchody. Cestou si napíšete obecnou konvoluci s maticí vah, zjistíte, co dělat za okrajem obrázku, zrychlíte kernel sdílenou pamětí a najdete ve fotce hrany.

Spusťte buňku. Otevře v Sáře nový dokument v sRGB s plátnem 1200 × 800 a v něm testovací fotku jako vrstvu `coffee`.
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

Funkce z úvodní lekce shrnuje její příloha. Sloupec Sekce říká, kde se funkce objeví poprvé.

**Kernel v buňce `%%gmacs`**

| Zápis | Co dělá | Sekce |
|---|---|---|
| `halo=32` na prvním řádku buňky | kernel smí číst až 32 pixelů za okrajem vrstvy | 1 |
| `uniform float weights[9]` | matice 3 × 3 jako mřížka políček pod buňkou, čtená po řádcích | 1 |
| `min`, `max`, `clamp` na `ivec2`, `floor` na `vec2` a převod `ivec2(...)` | po složkách | 3 |
| `kernel`, `image`, `params:` | celý kernel místo funkce `pixel`, s obrázky a parametry | 4 |
| `shared`, `barrier()`, `local_id`, `group_id` | sdílená paměť skupiny vláken, čekání na celou skupinu a čísla vlákna a skupiny | 4 |

**NumPy**

| Volání | Co dělá | Sekce |
|---|---|---|
| `np.pad(a, šířky, mode=...)` | pole rozšířené o okraje podle pravidla, `"edge"`, `"symmetric"`, `"wrap"` nebo `"constant"` | 1, 3, 5, 6 |
| `np.roll(a, posun, axis)` | pole posunuté dokola | 3 |
| `a[:, i:i + w]` | výřez pole, pohled na stejná data bez kopie | 1, 6 |
""")

md(r"""
## Cíl lekce

Buňka níže spustí hotový filtr. Jeho kód je schovaný, protože ho během lekce napíšete sami. Režim `Blur` rozmaže, `Sharpen` zaostří. Vyzkoušejte `Sharpen` s velkou `sigma` a velkým `amount`.
""")

code(GOAL, hidden=True)

# --- 1. Convolution ---------------------------------------------------------------
md(r"""
## 1. Konvoluce

Nová hodnota pixelu je vážený součet jeho okolí. Váhy tvoří malou matici, **jádro** (*kernel*, tady ve významu matice vah, ne programu na GPU). Pro jádro $w$ s poloměrem $r$:

$$g(x, y) = \sum_{j=-r}^{r} \sum_{i=-r}^{r} w(i, j)\, f(x + i, y + j)$$

@img(convolution_window.png, 900, Okno 3 krát 3 nad mřížkou pixelů, každý pixel okna vynásobený svou váhou a součet zapsaný do pixelu uprostřed)

Přísně vzato konvoluce jádro otáčí o 180°, $f(x - i, y - j)$, a vzorec nahoře je korelace. Pro souměrná jádra, jako je rozmazání, vyjdou stejně.

Nejjednodušší jádro má všechny váhy stejné, $1 / (2r + 1)^2$, a dá **průměr okolí** (*box blur*). Kernel níže čte okolí až 32 pixelů daleko. Slovo `halo=32` na prvním řádku buňky říká Sáře, kolik pixelů za okrajem vrstvy kernel čte. Bez něj čte jen 16 a dál dostane průhlednou černou.
""")

code(BOX)

md(r"""
### 🎯 Úkol 1: konvoluce s maticí

Napište kernel, který počítá konvoluci s libovolnou maticí 3 × 3. Parametr `uniform float weights[9]` se pod buňkou ukáže jako mřížka devíti políček, kernel ji dostane po řádcích: `weights[0]` až `weights[2]` je horní řádek. Výchozí hodnoty za `=` jsou zaostření, napsané po řádcích. Nedokončený kernel vrací původní pixel.

<details><summary>💡 Nápověda</summary>

1. Dva vnořené cykly přes řádek `j` a sloupec `i`, každý od 0 do 2.
2. Políčko `(i, j)` patří sousedovi s posunem `ivec2(i - 1, j - 1)` a jeho váha je `weights[j * 3 + i]`.
3. Sčítejte jen barvu, `.rgb`, a alfu vraťte z původního pixelu.
</details>

<details><summary>🔑 Řešení</summary>

""" + answer(SOLUTION_CONV) + r"""
</details>
""")

code(CONV, solve=[(CONV_STUB, SOLUTION_CONV)])

md(r"""
Zkuste do mřížky zadat další jádra. Jednička uprostřed a jinde nuly obrázek nezmění.

| Jádro | Mřížka po řádcích | Co dělá |
|---|---|---|
| průměr | všude 0.111 | rozmaže |
| zaostření | 0 −1 0 / −1 5 −1 / 0 −1 0 | zvýrazní rozdíl pixelu od sousedů |
| hrany | −1 −1 −1 / −1 8 −1 / −1 −1 −1 | nechá jen rozdíl od sousedů, plochy zčernají |
| reliéf (*emboss*) | −2 −1 0 / −1 1 1 / 0 1 2 | hrany světlé z jedné strany a tmavé z druhé |

@img(kernel_gallery.png, 900, Výřez fotky po konvoluci se čtyřmi jádry z tabulky: průměr, zaostření, hrany a reliéf)

### ✅ Kontrola

Vraťte do mřížky zaostření, nebo buňku kernelu spusťte znovu, a spusťte kontrolu. Porovná vrstvu `Convolution` s konvolucí v NumPy.
""")

code(f"""
img = photo.read()
w = np.array({SHARPEN}, dtype=np.float32).reshape(3, 3)
padded = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode="edge")
sharp = img.copy()
sharp[..., :3] = sum(w[j, i] * padded[j:j + {HEIGHT}, i:i + {WIDTH}, :3] for j in range(3) for i in range(3))
sara.check("Convolution", sharp)
""")

md(r"""
> **❓ Otázka**
> Součet vah průměru i zaostření je 1, součet vah hran je 0. Co součet vah říká o tom, co jádro udělá s jednobarevnou plochou?

<details><summary>🔑 Odpověď</summary>

Na ploše jsou všichni sousedé stejní, takže výsledek je barva plochy krát součet vah. Se součtem 1 plocha zůstane, jak byla, a jádro mění jen místa, kde se sousedé liší. Se součtem 0 plocha zčerná a zbydou jen změny, hrany. Proto se jádro rozmazání dělí součtem svých vah.
</details>
""")

# --- 2. Separable Gaussian --------------------------------------------------------
md(r"""
## 2. Gauss ve dvou průchodech

Průměr okolí dělá z jasných bodů hranaté skvrny, protože vzdálený soused váží stejně jako blízký. **Gaussovo rozmazání** váží souseda podle vzdálenosti:

$$w(i, j) = e^{-\frac{i^2 + j^2}{2\sigma^2}}$$

Váha ve vzdálenosti $3\sigma$ je $e^{-4.5}$, asi setina té prostřední, a vzdálenější váhy kernel vynechá. Rozmazání míchá světlo, a proto počítá v lineárním RGB, jako míchání barev v lekci o barevných prostorech.
""")

code(GAUSS_2D)

md(r"""
Kernel sčítá čtverec o straně $2r + 1$, s $r = \lceil 3\sigma \rceil$. Při $\sigma = 8$ je to 2401 sousedů na pixel. Gaussova váha se ale rozpadne na součin dvou:

$$e^{-\frac{i^2 + j^2}{2\sigma^2}} = e^{-\frac{i^2}{2\sigma^2}} \cdot e^{-\frac{j^2}{2\sigma^2}}$$

Jádro je tedy **oddělitelné** (*separable*): stačí rozmazat každý řádek a výsledek pak rozmazat po sloupcích. Dva průchody sečtou $2(2r + 1)$ sousedů místo $(2r + 1)^2$, při $\sigma = 8$ tedy 98 místo 2401.

@img(separable.png, 900, Dvourozměrné Gaussovo jádro jako součin sloupce a řádku a počet čtených sousedů obou postupů v závislosti na poloměru)

### 🎯 Úkol 2: první průchod

Napište první průchod, rozmazání po řádku. Výsledek nechte v lineárním RGB, druhý průchod ho čte tak, jak je, a do sRGB převede až výsledek. Vrstva si lineární hodnoty pamatuje, jen v Sáře vypadá tmavší. Nedokončený kernel jen převede pixel do lineárního RGB.

<details><summary>💡 Nápověda</summary>

1. Vezměte kernel `Gauss 2D` a nechte jen vnitřní cyklus, přes `dx`.
2. Váha závisí jen na `dx`.
3. Na konci vraťte součet dělený součtem vah, bez převodu do sRGB.
</details>

<details><summary>🔑 Řešení</summary>

""" + answer(SOLUTION_BLUR_X) + r"""
</details>
""")

code(BLUR_X, solve=[(BLUR_X_STUB, SOLUTION_BLUR_X)])

md(r"""
Druhý průchod je hotový, čte vrstvu `Blur X`. Posuvník prvního průchodu druhý průchod sám nespustí: po změně `sigma` v prvním průchodu nastavte stejnou hodnotu i ve druhém.
""")

code(BLUR_Y)

md(r"""
### ✅ Kontrola

Oba průchody dohromady mají dát totéž co `Gauss 2D`. Nechte všude `sigma` na 3, nebo ji nastavte všude stejně.
""")

code("""
sara.check("Blur Y", doc.layer("Gauss 2D").read())
""")

md(r"""
### Rychlost

Buňka změří oba postupy s hodnotami, které mají posuvníky právě nastavené.
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
> Nastavte `sigma` ve všech třech kernelech na 8 a měření spusťte znovu. Kolikrát se zpomalil kernel `Gauss 2D` a kolikrát dva průchody? Odpovídá to počtu sousedů?

<details><summary>🔑 Odpověď</summary>

Poloměr vzrostl z 9 na 24. Jeden průchod čte $(2r + 1)^2$ sousedů, $49^2 / 19^2 \approx 6.7$krát víc, dva průchody $2(2r + 1)$, jen $98 / 38 \approx 2.6$krát víc. Naměřené poměry tomu odpovídají jen zhruba. U kernelu s málo sousedy tvoří většinu času režie, spuštění a zápis vrstvy, a GPU čte sousedy z rychlé vyrovnávací paměti (*cache*), takže čtení navíc stojí málo. Dva průchody navíc celou vrstvu `Blur X` jednou zapíšou a znovu přečtou. Při malém poloměru proto dva průchody nemusí vyhrát. Před dalšími kontrolami vraťte `sigma` ve všech třech kernelech na 3.
</details>
""")

# --- 3. Edges ---------------------------------------------------------------------
md(r"""
## 3. Za okrajem obrázku

Pixel u okraje vrstvy nemá sousedy na všech stranách. Pravidel, co číst za okrajem, je několik:

| Pravidlo | Za okrajem je | V NumPy |
|---|---|---|
| Zero | černá | `np.pad(..., mode="constant")` |
| Clamp | nejbližší pixel na okraji, okraj se natáhne | `mode="edge"` |
| Mirror | obrázek zrcadlený podle okraje | `mode="symmetric"` |
| Wrap | protější okraj, obrázek se opakuje jako dlaždice | `mode="wrap"` |

@img(edge_modes.png, 900, Fotka posunutá doprava dolů a čtyři pravidla pro to, co se objeví za okrajem: černá, natažený okraj, zrcadlo a dlaždice)

Sára sama čte za okrajem plátna nejbližší pixel, tedy Clamp, dokud nejde dál než `halo`. Kernel níže fotku posune o `dx` a `dy` a za okrajem čte podle pravidla `mode`. Posun je jen nejnázornější případ, rozmazání u okraje čte za okraj stejně, jen méně daleko.

### 🎯 Úkol 3: pravidla okraje

Doplňte do `fetch` pravidla Clamp, Mirror a Wrap. Každé z pozice `p` mimo plátno spočítá pozici `q` uvnitř a vrátí `src(q)`. Pravidlo Zero je hotové jako příklad. Výchozí pravidlo je Wrap, a nedokončený kernel je proto celý černý, dokud funkci nedopíšete. Zero uvidíte, když `mode` přepnete.

<details><summary>💡 Nápověda</summary>

1. Clamp: `clamp` funguje i na `ivec2`, po složkách.
2. Mirror pro souřadnici $x$ v plátně šířky $W$: záporné $x$ se zrcadlí na $-1 - x$, $x \ge W$ na $2W - 1 - x$. Obojí jde napsat po složkách funkcemi `max` a `min`.
3. Wrap je zbytek po dělení, který je i pro záporné $x$ od 0 do $W - 1$. Operátor `%` v GLSL na záporných číslech výsledek nezaručuje. Pomůže $x - W \lfloor x / W \rfloor$, s dělením ve `float`.
</details>

<details><summary>🔑 Řešení</summary>

""" + answer(SOLUTION_FETCH) + r"""
</details>
""")

code(EDGES, solve=[(FETCH_STUB, SOLUTION_FETCH)])

md(r"""
### ✅ Kontrola

Vraťte `mode` na `Wrap` a posuvníky `dx` a `dy` na výchozí hodnoty, nebo buňku kernelu spusťte znovu. Posun dokola je v NumPy `np.roll`.
""")

code(f"""
sara.check("Shifted", np.roll(photo.read(), ({SHIFT[1]}, {SHIFT[0]}), axis=(0, 1)))
""")

md(r"""
> **❓ Otázka**
> Které pravidlo se hodí k rozmazání fotky, které k rozmazání dlaždice textury a proč je Zero u rozmazání skoro vždy špatně?

<details><summary>🔑 Odpověď</summary>

U fotky je nejlepší Mirror nebo Clamp: za okrajem pokračuje něco podobného tomu, co je u okraje, a okraj rozmazané fotky má správnou barvu. Clamp u velkého poloměru dělá pruhy z natažených pixelů, Mirror ne. Textura, která se má opakovat, potřebuje Wrap, jinak by se po rozmazání na švech dlaždic objevil skok. Zero přimíchá do okrajových pixelů černou, okraje rozmazané fotky ztmavnou. Fourierova transformace v lekci o frekvencích bere obrázek vždy jako dlaždici, tedy Wrap.
</details>
""")

# --- 4. Shared memory -------------------------------------------------------------
md(r"""
## 4. Sdílená paměť

V prvním průchodu čte každé vlákno $2r + 1$ pixelů řádku, a sousední vlákna čtou skoro tytéž pixely. Při $\sigma = 8$ přečte GPU každý pixel 49krát a 49krát ho převede ze sRGB do lineárního RGB. Vlákna jedné **skupiny** (*workgroup*) mají společnou rychlou **sdílenou paměť** (*shared memory*), kterou vidí jen ta skupina. Skupina si do ní řádek načte jednou, každé vlákno jeden nebo dva pixely, počká na ostatní a pak sčítá ze sdílené paměti. Tak to dělají knihovny pro zpracování obrazu na GPU.

@img(shared_tile.png, 900, Skupina 64 vláken načte do sdílené paměti svých 64 pixelů a 24 pixelů na každé straně, pak každé vlákno sečte 49 hodnot ze sdílené paměti)

Funkce `pixel` sdílenou paměť nemá, kernel je proto napsaný celý. Má řádek `kernel` se jménem, obrázky `src` a `dst`, parametry pod `params:` a vlastní velikost skupiny, 64 vláken v řádku. Číslo vlákna ve skupině je `local_id`, číslo skupiny `group_id` a `pixel` pozice vlákna v obrázku. `barrier()` počká, až všechna vlákna skupiny dojdou na totéž místo, jinak by vlákno mohlo číst z paměti, kam soused ještě nezapsal. Parametr celého kernelu nemá výchozí hodnotu, a tak ji dává první řádek buňky, `sigma=3`.

Celý kernel vidí obrázek včetně okraje `halo`, takže jeho `pixel` je posunutý o 32 proti pozici na plátně a `image_size(src)` je o 64 větší než plátno. Zápis do `dst` na stejný `pixel` přistane na správném místě.
""")

code(SHARED)

code("""
sara.check("Blur X shared", doc.layer("Blur X").read())
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

# --- 5. Sobel ---------------------------------------------------------------------
md(r"""
## 5. Hrany

Hrana je místo, kde se rychle mění světlost. Změnu měří **gradient**, vektor derivací ve směru $x$ a $y$. **Sobelův operátor** odhadne derivace dvěma jádry 3 × 3, rozdílem pravého a levého sloupce a spodního a horního řádku, prostřední řádek a sloupec s dvojnásobnou vahou:

$$G_x = \begin{pmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{pmatrix} \qquad G_y = \begin{pmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{pmatrix}$$

Délka gradientu $\sqrt{g_x^2 + g_y^2}$ je síla hrany, jeho směr míří od tmavé strany ke světlé, kolmo na hranu.

@img(sobel.png, 900, Sobelova jádra Gx a Gy, gradient na hraně kolmo na ni a barevné kolo směrů, které kernel používá)

### 🎯 Úkol 4: Sobel

Spočítejte gradient světlosti OKLab L, `light(p)`, v proměnné `g`: `g.x` je součet s jádrem $G_x$, `g.y` s jádrem $G_y$. Funkce `shade` je hotová a obarví hranu podle směru, když zapnete `direction`. Nedokončený kernel vrací černou.

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
### ✅ Kontrola

Nechte `direction` vypnutý a `gain` na 2 a spusťte obě buňky. První obsahuje převody z lekce o barevných prostorech, světlost `lightness`, `rgba` a váhy Gaussova jádra `gaussian_1d`, které budou potřeba v další sekci. Druhá porovná sílu hran se Sobelem v NumPy.
""")

code(HELPERS_NP)

code("""
L = np.pad(lightness(photo.read()), 1, mode="edge")
h, w = L.shape[0] - 2, L.shape[1] - 2
gx = sum((i - 1) * (2 - abs(j - 1)) * L[j:j + h, i:i + w] for j in range(3) for i in range(3))
gy = sum((j - 1) * (2 - abs(i - 1)) * L[j:j + h, i:i + w] for j in range(3) for i in range(3))
sara.check("Sobel", rgba(np.clip(2.0 * np.hypot(gx, gy), 0, 1)))
""")

md(r"""
> **❓ Otázka**
> Zapněte `direction`. Proč mají dvě strany tenké čáry, třeba okraje lžičky, opačné barvy?

<details><summary>🔑 Odpověď</summary>

Barva je směr gradientu, od tmavší strany ke světlejší. Tenká světlá čára má tmavé okolí na obou stranách: na jedné straně světlost roste, na druhé klesá, a gradienty míří proti sobě. Opačné směry dostanou na barevném kole opačné barvy. Detektory hran, například Cannyho, to využívají: hranu ztenčí na jeden pixel tak, že nechají jen místa, kde je síla největší ve směru gradientu.
</details>
""")

# --- 6. NumPy and Krita -----------------------------------------------------------
md(r"""
## 6. Do Krity

Filtr do Krity je v NumPy. NumPy nemá vlákna na pixel, ale umí celé pole najednou: posunutý obrázek je **výřez** rozšířeného pole, `p[:, i:i + w]`, pohled na tatáž data bez kopie. Vážený součet $2r + 1$ výřezů je jeden průchod rozmazání.

@img(numpy_shifts.png, 760, Řádek rozšířený o okraje a pět výřezů posunutých o jeden pixel, každý vynásobený svou vahou a sečtený)

### 🎯 Úkol 5: rozmazání v NumPy

Napište `blur(a, sigma)`: rozmazání po řádcích a pak po sloupcích, okraj podle pravidla Clamp. Váhy dává `gaussian_1d(sigma)`, pole délky $2r + 1$. Nedokončená funkce pole nemění. Kontrola porovnává s vrstvou `Gauss 2D`, takže kernel `Gauss 2D` má mít `sigma` 3.

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
print(f"NumPy blur: {{(time.perf_counter() - start) * 1000:.1f}} ms")
sara.check("Gauss 2D", rgba(linear_to_srgb(soft)))
""", solve=[(BLUR_NP_GIVEN, SOLUTION_BLUR_NP)])

md(r"""
**Zaostření** (*unsharp mask*) přičte k obrázku jeho rozdíl od rozmazaného obrázku, $f + a\,(f - \text{blur}(f))$. Rozdíl jsou detaily menší než $\sigma$ a `amount` $a$ říká, kolikrát je zesílit. Název je z fotokomory, kde se k negativu přikládala rozmazaná, tedy neostrá (*unsharp*), maska.

### 🎯 Úkol 6: filtr

Napište `apply` s režimy `Blur` a `Sharpen`, oba v lineárním RGB. Zaostření může dát záporné světlo, ořízněte ho na nulu. Nedokončená funkce vrací obrázek beze změny.

<details><summary>💡 Nápověda</summary>

1. Převeďte barvu do lineárního RGB a rozmažte ji funkcí `blur`.
2. `Blur` vrátí rozmazaný obrázek, `Sharpen` obrázek plus `amount` krát rozdíl, oříznutý `np.maximum(..., 0.0)`.
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
### ✅ Kontrola

Porovnání s filtrem „Goal“ z úvodu s jeho výchozími hodnotami. Pokud jste s nimi hýbali, spusťte buňku „Goal“ znovu.
""")

code(f"""
defaults = {DEFAULTS}
sara.check("Goal", apply(photo.read(), **defaults))
""")

md(r"""
### 🎯 Úkol 7: filtr v Kritě

Vložte filtr do `pga_filter/effect.py` s `TITLE = "Blur and sharpen"` a vyzkoušejte ho na vlastní fotce. Porovnejte rychlost s Gaussovým rozmazáním, které má Krita vestavěné.

<details><summary>💡 Nápověda</summary>

1. Do `effect.py` patří pomocné funkce, vaše `blur`, `PARAMS` a `apply`. Řádek `import numpy as np` nechte nahoře.
2. Náhled počítá jen malou kopii, pomalé je až Apply na celé fotce. Zkuste filtr nejdřív na výběru. Ve zmenšeném náhledu „Whole region, scaled down“ vypadá rozmazání silnější, skutečnou sílu ukáže „Centre at full size“.
</details>

> **❓ Otázka**
> Fotka s průhledným okolím, třeba vystřižený předmět na průhledné vrstvě, má po rozmazání kolem předmětu tmavý lem. Proč, a jak by se mu dalo předejít?

<details><summary>🔑 Odpověď</summary>

Průhledné pixely mají nějakou barvu, obvykle černou, a rozmazání ji přimíchá k barvě předmětu, i když ji nikdo nevidí. Pomůže rozmazávat barvu vynásobenou alfou (*premultiplied alpha*), rozmazat i alfu a výslednou barvu alfou zase vydělit. Průhledný pixel pak k barvě nepřidá nic.
</details>
""")

# --- Summary ------------------------------------------------------------------
md(r"""
## Shrnutí

- Konvoluce je vážený součet okolí. Součet vah 1 zachová plochy, součet 0 nechá jen změny.
- Gaussovo jádro je oddělitelné: dva průchody po řádcích a sloupcích čtou $2(2r + 1)$ sousedů místo $(2r + 1)^2$.
- Rozmazání míchá světlo, a proto počítá v lineárním RGB. Mezivýsledek může v lineárním RGB zůstat.
- Za okrajem obrázku je nutné něco zvolit: Zero ztmaví okraje, Clamp je natáhne, Mirror zrcadlí a Wrap opakuje obrázek jako dlaždici.
- Sdílená paměť skupiny ušetří opakované čtení stejných pixelů. Vyplatí se u velkých jader s drahým čtením.
- Sobelův operátor odhadne gradient světlosti. Jeho délka je síla hrany, směr míří kolmo na hranu.
- NumPy počítá konvoluci jako vážený součet posunutých výřezů celého pole.

### Co jsme vynechali

- **Rozmazání s proměnným poloměrem.** Hloubka ostrosti nebo rozmazání podle masky mění $\sigma$ pixel od pixelu a jádro pak není oddělitelné.
- **Rychlé aproximace.** Několik průměrů za sebou se blíží Gaussovi, a průměr jde spočítat v čase nezávislém na poloměru, posuvným součtem. Kawaseho rozmazání a pyramida zmenšených obrázků jsou rychlé způsoby, jak hry dělají rozmazání pro záři.
- **Rozmazání, které zachová hrany.** Bilaterální filtr váží souseda i podle toho, jak se liší barvou, a hrany nerozmaže. Mezi taková rozmazání patří i Kuwaharův filtr z lekce o stylizaci.
- **Cannyho detektor hran.** Gaussovo rozmazání, Sobel, ztenčení hran na jeden pixel a dvě prahové hodnoty dají dohromady čisté čáry místo šedých pruhů.
- **Konvoluce přes Fourierovu transformaci.** Pro velká jádra je rychlejší násobit spektra, jak ukáže lekce o frekvencích.

### Bonusové úkoly

1. **Ostré světlo.** Rozmažte fotku s několika jasnými body jednou v sRGB a jednou v lineárním RGB a porovnejte světlé skvrny. Která verze vypadá jako rozostřený objektiv?
2. **Průměr v konstantním čase.** Napište průměr okolí v NumPy jako rozdíl kumulativních součtů, `np.cumsum`, takže čas nezávisí na poloměru. Třikrát za sebou dá skoro Gausse.
3. **Sdílený druhý průchod.** Napište druhý průchod se sdílenou pamětí, se skupinou 1 × 64 vláken ve sloupci.
4. **Ostření jen na hranách.** Použijte sílu hrany ze Sobela jako masku zaostření, aby se nezesílil šum na plochách.
""")

md(r"""
### Nápady na semestrální práci

- Rozostření podle masky: plugin rozmaže vrstvu podle světlosti druhé vrstvy nebo výběru, jako tilt-shift fotografie.
- Detektor hran s volbou operátoru (Sobel, Scharr, Prewitt) a výstupem jako čárová kresba nebo maska výběru.
- Ostření s ochranou hran a šumu, s náhledem masky, kde se ostří.
""")


def build():
    lesson_writer.write(cells, OUT, "l3")


if __name__ == "__main__":
    build()
