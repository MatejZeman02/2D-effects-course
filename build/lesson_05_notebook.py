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
BLURS = f"""\
%%gmacs "chelsea" -> "Blurs" halo=32
import linear_srgb_color_space
import oklab_color_space

uniform float sigma: hint_range(0.3, 4) = {SIGMA}
uniform float k: hint_range(1.1, 3) = {K}


def light(ivec2 p) -> float:
    return linear_srgb_to_oklab(srgb_to_linear_srgb(src(p).rgb)).x


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
            float d2 = float(dx * dx + dy * dy)
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
    vec4 b = src(at)
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
    vec4 b = src(at)
{XDOG_STUB}"""

TOON_STUB = """\
    # TODO: the lightness rounded to levels steps, then all of lab darkened by the ink"""

SOLUTION_TOON = """\
    float n = float(levels - 1)
    lab.x = floor(clamp(lab.x, 0.0, 1.0) * n + 0.5) / n
    lab *= ink"""

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


@workgroup_size(8, 8)
def toon():
    vec3 lab = linear_srgb_to_oklab(srgb_to_linear_srgb(image_load(src, pixel).rgb))
    float ink = image_load(lines, pixel).r
{TOON_STUB}
    vec3 rgb = clamp(oklab_to_linear_srgb(lab), vec3(0.0), vec3(1.0))
    image_store(dst, pixel, vec4(linear_srgb_to_srgb(rgb), 1.0))"""

QUADRANT_STUB = """\
    # TODO: the mean colour of the (radius + 1)^2 pixels, and the variance of their luminance
    return vec4(srgb_to_linear_srgb(src(at).rgb), 1.0)"""

SOLUTION_QUADRANT = """\
    vec3 total = vec3(0.0)
    float s1 = 0.0
    float s2 = 0.0
    for dy in range_inclusive(0, radius):
        for dx in range_inclusive(0, radius):
            vec3 c = srgb_to_linear_srgb(src(at + towards * ivec2(dx, dy)).rgb)
            float l = dot(c, vec3(0.2126, 0.7152, 0.0722))
            total += c
            s1 += l
            s2 += l * l
    float n = float((radius + 1) * (radius + 1))
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

TOON_NP = '''\
def toon(img, ink, levels):
    """img's colours with the lightness rounded to levels steps and darkened by ink, (h, w, 3) in sRGB."""
    lab = oklab(srgb_to_linear(img[..., :3]))
    n = levels - 1
    lab[..., 0] = np.floor(np.clip(lab[..., 0], 0, 1) * n + 0.5) / n
    lab *= ink[..., None]
    return linear_to_srgb(np.clip(oklab_to_linear(lab), 0, 1))'''

XDOG_NP_GIVEN = f'''\
def xdog(L, sigma, p, epsilon, phi, k={K}):
    """XDoG line art of the lightness L, (h, w), from 0 (ink) to 1 (paper)."""
    # TODO: the two blurs, S sharpened by p, then the soft threshold
    return np.ones_like(L)'''

SOLUTION_XDOG_NP = f'''\
def xdog(L, sigma, p, epsilon, phi, k={K}):
    """XDoG line art of the lightness L, (h, w), from 0 (ink) to 1 (paper)."""
    g1 = blur(L[..., None], sigma)[..., 0]
    g2 = blur(L[..., None], k * sigma)[..., 0]
    s = (1 + p) * g1 - p * g2
    return np.where(s >= epsilon, 1.0, 1.0 + np.tanh(phi * (s - epsilon)))'''

PARAMS_NP = f'''\
PARAMS = {{
    "style": ["Toon", "Lines"],
    "sigma": ({SIGMA}, 0.3, 4.0),
    "p": ({P}, 0.0, 100.0),
    "epsilon": ({EPSILON}, 0.0, 1.0),
    "phi": ({PHI}, 0.0, 100.0),
    "levels": ({LEVELS}, 2, 12),
}}'''

APPLY_GIVEN = '''\
def apply(img, style, sigma, p, epsilon, phi, levels):
    out = img.copy()
    # TODO: the lines alone, or the toon look with the lines over it
    return out'''

SOLUTION_APPLY = '''\
def apply(img, style, sigma, p, epsilon, phi, levels):
    out = img.copy()
    ink = xdog(lightness(img), sigma, p, epsilon, phi)
    if style == "Lines":
        out[..., :3] = ink[..., None]
    else:
        out[..., :3] = toon(img, ink, levels)
    return out'''

GOAL = ("# The filter of section 5, hidden because you write it yourself.\n"
        "def _goal():\n"
        + textwrap.indent("\n\n".join(piece.replace("\n\n\n", "\n\n") for piece in
                                        [HELPERS_NP, TOON_NP, SOLUTION_XDOG_NP, SOLUTION_APPLY]), "    ")
        + "\n\n    return apply\n\n\n"
        + PARAMS_NP
        + '\n\nsara.live(_goal(), PARAMS, source="chelsea", target="Goal")')

DEFAULTS = (f'{{"style": "Toon", "sigma": {SIGMA}, "p": {P}, "epsilon": {EPSILON}, '
            f'"phi": {PHI}, "levels": {LEVELS}}}')


def answer(snippet, lang="python"):
    """A solution block's code fence."""
    return f"```{lang}\n{snippet}\n```"


# --- Title --------------------------------------------------------------------
md(r"""
# Lekce 5: Stylizace, čárová kresba a kreslený vzhled

Na konci lekce budete mít v Kritě filtr, který z fotky udělá kresbu tuší, nebo kreslený obrázek s plochými barvami a obrysy. Postavíte ho z několika průchodů za sebou: dvě rozmazání z lekce 3, jejich rozdíl, měkký práh a posterizace z lekce 1. Navíc napíšete Kuwaharův filtr, který z fotky udělá malbu.

Spusťte buňku. Otevře v Sáře nový dokument v sRGB s plátnem 902 × 600 a v něm testovací fotku jako vrstvu `chelsea`.
""")

code(f"""
import sys
sys.path.insert(0, "lib")            # sara.py and its helpers live in lib/
import time
import numpy as np
import sara

doc, layer, pixels = sara.init(width={WIDTH}, height={HEIGHT})  # a canvas the size of the photo
photo = doc.import_image("imgs/5/chelsea.png")         # a layer called "chelsea"
""")

md(r"""
### Nové funkce v této lekci

Funkce z úvodní lekce shrnuje její příloha. Sloupec Sekce říká, kde se funkce objeví poprvé.

**Kernel v buňce `%%gmacs`**

| Zápis | Co dělá | Sekce |
|---|---|---|
| `tanh(x)` | hyperbolický tangens, plynulý schod od −1 do 1 | 2 |
| `image lines: readonly` a `lines=XDoG` na prvním řádku | druhý vstup celého kernelu, vrstva `XDoG` pod jménem `lines` | 3 |
| `int levels` pod `params:` a `levels=5` na prvním řádku | celočíselný parametr celého kernelu a jeho hodnota | 3 |

**NumPy**

| Volání | Co dělá | Sekce |
|---|---|---|
| `np.tanh(a)` | hyperbolický tangens po prvcích | 5 |
""")

md(r"""
## Cíl lekce

Buňka níže spustí hotový filtr. Jeho kód je schovaný, protože ho během lekce napíšete sami. Režim `Toon` dá kreslený vzhled, `Lines` jen čáry. Zkuste posuvníky `epsilon` a `p`: první říká, kolik obrázku zůstane bílé, druhé, jak silné jsou obrysy.
""")

code(GOAL, hidden=True)

# --- 1. DoG -----------------------------------------------------------------------
md(r"""
## 1. Rozdíl dvou rozmazání

Rozmazání z lekce 3 je dolní propust: nechá velké tvary a odstraní detaily menší než $\sigma$. Dvě rozmazání s různou $\sigma$ odstraní různě velké detaily, a jejich **rozdíl** (*difference of Gaussians*, DoG) nechá jen detaily mezi oběma velikostmi. Je to **pásmová propust** (*band pass*): ve spektru z lekce 4 ztlumí nízké i vysoké frekvence a nechá pásmo mezi nimi:

$$D = G_\sigma * L - G_{k\sigma} * L, \qquad k \approx 1.6$$

Na hraně je $D$ kladné na světlé straně a záporné na tmavé, uprostřed plochy nula. Podobně vidí hrany i sítnice: buňky reagují na rozdíl středu a okolí, a Marr a Hildreth z toho v roce 1980 udělali detektor hran.

@img(dog_profile.png, 900, Dvě Gaussovy křivky různé šířky, jejich rozdíl ve tvaru mexického klobouku a odezva rozdílu na schodovou hranu se zákmity na obou stranách)

Proč rozdíl dvou rozmazání najde hrany? Je to skoro **Laplaceův operátor** rozmazaného obrázku, druhá derivace z lekce o konvoluci. Rozmazání s rostoucím $\sigma$ se řídí rovnicí vedení tepla: změna Gaussova jádra s $\sigma$ je jeho Laplace, $\partial G_\sigma / \partial \sigma = \sigma\, \nabla^2 G_\sigma$. Rozdíl pro blízká $\sigma$ a $k\sigma$ je derivace krát krok:

$$G_{k\sigma} - G_\sigma \approx (k\sigma - \sigma)\, \frac{\partial G_\sigma}{\partial \sigma} = (k - 1)\, \sigma^2\, \nabla^2 G_\sigma$$

$$D \approx -(k - 1)\, \sigma^2\, \nabla^2 (G_\sigma * L)$$

DoG je tedy záporný Laplace rozmazané světlosti (*Laplacian of Gaussian*, LoG), až na násobek. Rozmazání napřed potlačí šum, který by druhá derivace jinak zesílila nejvíc. Marr a Hildreth hledali hrany jako místa, kde LoG prochází nulou, a s $k \approx 1.6$ je DoG tvarem LoG nejblíž.

Kernel níže spočítá obě rozmazání světlosti OKLab L najednou a uloží je do jedné vrstvy: $G_\sigma$ do červené, $G_{k\sigma}$ do zelené a samotnou světlost do modré. Další kernely ji čtou jako tři čísla, ne jako barvu. Na plochách jsou všechna tři čísla skoro stejná, a tak vrstva vypadá šedě, s barevnými lemy u hran.
""")

code(BLURS)

md(r"""
### 🎯 Úkol 1: DoG

Spočítejte rozdíl červené a zelené složky vrstvy `Blurs`, vynásobte ho `gain` a přičtěte k šedé 0.5, aby byly vidět kladné i záporné hodnoty. Nedokončený kernel vrací světlost.

<details><summary>💡 Nápověda</summary>

`b.r` je $G_\sigma$, `b.g` je $G_{k\sigma}$. Výsledek je šedý, `vec3` ze tří stejných čísel.
</details>

<details><summary>🔑 Řešení</summary>

""" + answer(SOLUTION_DOG) + r"""
</details>
""")

code(DOG, solve=[(DOG_STUB, SOLUTION_DOG)])

md(r"""
### ✅ Kontrola

Pomocné funkce obsahují převody z lekce o barevných prostorech a rozmazání `blur` z lekce 3. Kontrola porovná vrstvu `DoG` s výpočtem v NumPy, s výchozími hodnotami posuvníků.
""")

code(HELPERS_NP)

code(f"""
L = lightness(photo.read())[..., None]
dog = blur(L, {SIGMA}) - blur(L, {K} * {SIGMA})
sara.check("DoG", rgba(0.5 + 10.0 * dog[..., 0]))
""")

md(r"""
> **❓ Otázka**
> Zvětšete `sigma` v buňce `Blurs` na 3 a pak ji zmenšete na 0.5. Jak se změní, které hrany DoG najde? Kernel `DoG` po změně spusťte znovu. Nakonec spusťte buňku `Blurs` znovu, aby měla zase výchozí `sigma` 1.5.

<details><summary>🔑 Odpověď</summary>

S velkou $\sigma$ zůstanou jen obrysy velkých tvarů, hlava a uši, a srst zmizí. S malou $\sigma$ najde DoG i jednotlivé chlupy a šum. $\sigma$ volí velikost detailu, který se počítá jako hrana, a DoG je proto pásmová propust.
</details>
""")

# --- 2. XDoG ----------------------------------------------------------------------
md(r"""
## 2. XDoG: čáry jako tuší

DoG je šedý obrázek kolem nuly, kresba tuší ale potřebuje černou na bílé. Winnemöller, Kyprianidis a Olsen v roce 2012 rozšířili DoG o dvě úpravy (*eXtended DoG*, XDoG). Nejdřív přičtou zesílený rozdíl k rozmazanému obrázku, což je zaostření z lekce 3:

$$S = (1 + p)\, G_\sigma * L - p\, G_{k\sigma} * L = G_\sigma * L + p\, D$$

Pak $S$ projde **měkkým prahem**: od $\varepsilon$ výš je papír bílý, pod $\varepsilon$ klesá plynulý schod k černé, strmý podle $\varphi$:

$$T(S) = \begin{cases} 1 & S \ge \varepsilon \\ 1 + \tanh\!\big(\varphi\,(S - \varepsilon)\big) & S < \varepsilon \end{cases}$$

@img(xdog_curve.png, 760, Měkký práh T podle S pro několik strmostí phi, od plynulého přechodu po ostrý schod)

S velkým $\varphi$ je práh ostrý a obrázek černobílý jako dřevoryt. S malým $\varphi$ zůstanou v tmavých místech šedé tóny jako kresba uhlem.

### 🎯 Úkol 2: XDoG

Napište kernel podle vzorců. Nedokončený kernel vrací světlost.

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
### ✅ Kontrola

Porovnejte vrstvu `XDoG` s obrázkem níže, levým horním, který má výchozí hodnoty. Číselná kontrola přijde v sekci 5, až budete mít XDoG i v NumPy. Pak zkuste posuvníky: ostatní obrázky ukazují, co dělají.

@img(xdog_params.png, 900, Šest variant čárové kresby kočky s různými hodnotami sigma, p, epsilon a phi)

> **❓ Otázka**
> Co se stane s $p = 0$? A proč obrázek s $p = 60$ a $\varepsilon = 0.9$ vypadá jako negativ?

<details><summary>🔑 Odpověď</summary>

S $p = 0$ je $S = G_\sigma * L$, rozmazaná světlost, a XDoG je jen práh světlosti: tmavé plochy černé, světlé bílé, žádné obrysy. Velké $p$ zesílí rozdíl, takže na světlé straně hran je $S$ mnohem větší než 1 a na tmavé záporné. S $\varepsilon = 0.9$ je skoro každá plocha pod prahem a zčerná. Bílé zůstanou jen světlé strany hran, které $p$ vytáhlo nad $\varepsilon$: bílé čáry na černém, jako negativ kresby.
</details>
""")

# --- 3. Toon ----------------------------------------------------------------------
md(r"""
## 3. Kreslený vzhled

Kreslený vzhled (*toon shading*) napodobuje kreslený film: málo odstínů a černé obrysy. Odstíny dá posterizace světlosti z lekce 1, obrysy XDoG. Kernel teď čte **dvě vrstvy**: fotku jako `src` a čáry jako `lines`. Funkce `pixel` umí jen jeden vstup, a tak je kernel napsaný celý, jako sdílený průchod v lekci 3. Řádek `image lines: readonly` přidá druhý obrázek a `lines=XDoG` na prvním řádku buňky říká, která vrstva to je.

@img(toon_steps.png, 900, Fotka kočky, její posterizovaná světlost, čáry XDoG a obojí dohromady jako kreslený obrázek)

### 🎯 Úkol 3: toon

Zaokrouhlete světlost `lab.x` na `levels` úrovní a pak vynásobte celé `lab` hodnotou čáry `ink`. Barva se tak u čáry ztmaví i odbarví. Nedokončený kernel vrací fotku.

<details><summary>💡 Nápověda</summary>

1. Posterizace z lekce 1 s prahem 0.5: $\lfloor L\,(n - 1) + 0.5 \rfloor / (n - 1)$.
2. `levels` je `int`, převeďte ho na `float`.
3. `lab *= ink` vynásobí všechny tři složky.
</details>

<details><summary>🔑 Řešení</summary>

""" + answer(SOLUTION_TOON) + r"""
</details>
""")

code(TOON, solve=[(TOON_STUB, SOLUTION_TOON)])

md(r"""
### ✅ Kontrola

Funkce `toon` je totéž v NumPy. Budete ji potřebovat v sekci 5.
""")

code(TOON_NP + f"""


ink = doc.layer("XDoG").read()[..., 0]
sara.check("Toon", rgba(toon(photo.read(), ink, {LEVELS})))
""")

md(r"""
Posterizace zaokrouhluje, takže pixel na hranici dvou úrovní se na GPU a v NumPy může zaokrouhlit různě. Desetiny procenta jsou v pořádku.

> **❓ Otázka**
> Srst kočky je po posterizaci zrnitá, plná skvrn jedné a druhé úrovně. Proč, a co by pomohlo?

<details><summary>🔑 Odpověď</summary>

Světlost srsti se mění pixel od pixelu kolem hranice dvou úrovní, a každý pixel se zaokrouhlí jinam. Pomůže fotku před posterizací vyhladit tak, aby plochy byly hladké a hrany zůstaly ostré. Gaussovo rozmazání by rozmazalo i hrany, a tak přichází Kuwaharův filtr.
</details>
""")

# --- 4. Kuwahara ------------------------------------------------------------------
md(r"""
## 4. Kuwahara: fotka jako malba

Kuwaharův filtr z roku 1976 vyhladí plochy a hrany nechá ostré. Kolem pixelu vezme čtyři čtverce, každý s pixelem v jednom rohu. Spočítá v každém průměrnou barvu a rozptyl jasu, a pixel dostane průměr čtverce s **nejmenším rozptylem**, toho nejklidnějšího. U hrany je to čtverec, který leží celý na jedné straně, a hrana se proto nerozmaže. Výsledek vypadá jako olejomalba, s plochými tahy štětce.

@img(kuwahara.png, 900, Čtyři čtverce kolem pixelu na hraně, ten s nejmenším rozptylem zvýrazněný, a výřez fotky před filtrem a po něm)

Rozptyl je průměr čtverců minus čtverec průměru:

$$\operatorname{var} = \frac{1}{n}\sum l^2 - \Big(\frac{1}{n}\sum l\Big)^2$$

### 🎯 Úkol 4: Kuwahara

Doplňte funkci `quadrant`. Čtverec má stranu `radius + 1` a vede od `at` ve směru `towards`: pixel `at + towards * ivec2(dx, dy)` pro `dx` a `dy` od 0 do `radius`. Vrací průměrnou barvu v lineárním RGB a ve složce `.a` rozptyl jasu $l = 0.2126\,R + 0.7152\,G + 0.0722\,B$. Funkce `calmer` a `pixel` jsou hotové. Nedokončená funkce vrací pixel samotný.

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
### ✅ Kontrola

Totéž v NumPy, se čtyřmi čtverci ve stejném pořadí jako ve funkci `pixel`. Při shodném rozptylu dvou čtverců může GPU zvolit jiný čtverec než NumPy, takže desetiny procenta jsou v pořádku.
""")

code(f"""
r = {RADIUS}
lin = srgb_to_linear(photo.read()[..., :3])
lum = lin @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
h, w = lum.shape
P = np.pad(lin, ((r, r), (r, r), (0, 0)), mode="edge")
Q = np.pad(lum, r, mode="edge")
best, best_var = np.zeros_like(lin), np.full((h, w), np.inf)
for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
    total, s1, s2 = np.zeros_like(lin), np.zeros((h, w)), np.zeros((h, w))
    for dy in range(r + 1):
        for dx in range(r + 1):
            y, x = r + sy * dy, r + sx * dx
            total += P[y:y + h, x:x + w]
            s1 += Q[y:y + h, x:x + w]
            s2 += Q[y:y + h, x:x + w] ** 2
    n = (r + 1) ** 2
    var = s2 / n - (s1 / n) ** 2
    calmer = var < best_var
    best[calmer], best_var[calmer] = total[calmer] / n, var[calmer]
sara.check("Kuwahara", rgba(linear_to_srgb(best)))
""")

md(r"""
Teď kreslený vzhled podruhé, z malby místo z fotky. V buňce kernelu `Toon` změňte první řádek na

```
%%gmacs "Kuwahara" -> "Toon" lines=XDoG levels=5
```

a spusťte ji znovu. Zkuste také `Blurs` z vrstvy `Kuwahara`, aby i čáry byly klidnější: první řádek `%%gmacs "Kuwahara" -> "Blurs" halo=32` a pak znovu `XDoG` a `Toon`. Tak vypadá řetěz průchodů, *pipeline*: každý kernel čte vrstvu, kterou zapsal předchozí.

@img(pipeline.png, 900, Řetěz vrstev lekce: fotka, Kuwahara, rozmazání, XDoG a toon, se šipkami od vrstvy ke kernelu, který ji čte)

> **❓ Otázka**
> Zvětšete `radius` na 8. Proč se objeví hranaté skvrny, a proč tak Kuwaharův filtr v praxi skoro nikdo nepoužívá?

<details><summary>🔑 Odpověď</summary>

Čtverce mají rovné strany a pevné směry, takže výsledek se skládá z kousků čtverců, hlavně tam, kde se nejklidnější čtverec mění skokem. Varianty, které se používají, mění tvar oblasti: kruhové výseče místo čtverců s plynulými vahami (*generalised Kuwahara*) a výseče protažené podél hrany (*anisotropic Kuwahara*), jak to ukazuje Acerola ve videu o Kuwaharově filtru.
</details>
""")

# --- 5. NumPy and Krita -----------------------------------------------------------
md(r"""
## 5. Do Krity

Filtr do Krity spojí XDoG a toon v NumPy. Rozmazání `blur` a funkci `toon` už máte.

### 🎯 Úkol 5: XDoG v NumPy

Napište `xdog(L, sigma, p, epsilon, phi)`. `L` je pole `(h, w)`, `blur` chce `(h, w, c)`, takže světlost rozšiřte o osu a výsledek o ni zase zmenšete. Nedokončená funkce vrací bílý papír.

Buňka pod funkcí porovná výsledek s vrstvou `XDoG`. Pokud jste na konci sekce 4 přepnuli `Blurs` na `Kuwahara` nebo hýbali posuvníky, vraťte první řádek `Blurs` na `%%gmacs "chelsea" -> "Blurs" halo=32` a spusťte znovu `Blurs` a pak `XDoG`.

<details><summary>💡 Nápověda</summary>

1. `blur(L[..., None], sigma)[..., 0]` je rozmazaná světlost `(h, w)`.
2. `np.where(s >= epsilon, 1.0, ...)` vybere bílou nebo měkký práh po prvcích. Spočítá obě větve všude, ale to nevadí.
</details>

<details><summary>🔑 Řešení</summary>

""" + answer(SOLUTION_XDOG_NP) + r"""
</details>
""")

code(XDOG_NP_GIVEN + f"""


start = time.perf_counter()
lines = xdog(lightness(photo.read()), {SIGMA}, {P}, {EPSILON}, {PHI})
print(f"NumPy XDoG: {{(time.perf_counter() - start) * 1000:.1f}} ms")
sara.check("XDoG", rgba(lines))
""", solve=[(XDOG_NP_GIVEN, SOLUTION_XDOG_NP)])

md(r"""
### 🎯 Úkol 6: filtr

Napište `apply`. Styl `Lines` vrátí čáry jako šedý obrázek, `Toon` posterizaci s čarami přes funkci `toon`. Nedokončená funkce vrací obrázek beze změny.

<details><summary>💡 Nápověda</summary>

1. `ink = xdog(lightness(img), sigma, p, epsilon, phi)`.
2. Šedý obrázek do tří kanálů: `ink[..., None]` se rozšíří sám.
3. Alfa zůstává z `img`.
</details>

<details><summary>🔑 Řešení</summary>

""" + answer(SOLUTION_APPLY) + r"""
</details>
""")

code(PARAMS_NP + "\n\n\n" + APPLY_GIVEN + """


sara.live(apply, PARAMS, source="chelsea", target="Filter NumPy")
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

Vložte filtr do `pga_filter/effect.py` s `TITLE = "XDoG line art"` a vyzkoušejte ho na portrétu, na krajině a na vlastní kresbě. Na velkých fotkách zvětšete `sigma`, protože detaily mají víc pixelů.

<details><summary>💡 Nápověda</summary>

Do `effect.py` patří pomocné funkce (převody, `blur`, `toon`), vaše `xdog`, `PARAMS` a `apply`. Řádek `import numpy as np` nechte nahoře.
</details>
""")

# --- Summary ------------------------------------------------------------------
md(r"""
## Shrnutí

- Rozdíl dvou Gaussových rozmazání je pásmová propust: nechá detaily mezi dvěma velikostmi a na hranách dá kladné a záporné hodnoty.
- DoG je skoro záporný Laplace rozmazaného obrázku (LoG), protože Gauss se s rostoucím $\sigma$ mění podle rovnice vedení tepla.
- XDoG k rozmazané světlosti přičte zesílený rozdíl a výsledek projde měkkým prahem. Čtyři čísla, $\sigma$, $p$, $\varepsilon$ a $\varphi$, nastaví vzhled od kresby tuší po kresbu uhlem.
- Kreslený vzhled je posterizovaná světlost krát čáry. Celý kernel čte dvě vrstvy, druhou přes `image ...: readonly` a jméno na prvním řádku buňky.
- Kuwaharův filtr vyhladí plochy a nechá hrany, protože bere průměr nejklidnějšího ze čtyř čtverců.
- Stylizace je řetěz průchodů: každý kernel zapíše vrstvu, kterou čte další, a každý článek jde vyměnit.

### Co jsme vynechali

- **Čáry podél hran.** Rozmazání podél směru hrany (*flow-based DoG*, s polem směrů *edge tangent flow*) spojí přerušované čáry do plynulých tahů. Směr hrany dává Sobelův operátor z lekce o konvoluci.
- **Anizotropní Kuwahara.** Výseče protažené podél hrany místo čtverců. Výsledek vypadá jako opravdová malba štětcem.
- **Šrafování a půltóny.** Světlost se dá převést na šrafy, tečky tiskového rastru nebo znaky ASCII. Znaky potřebují obrázek s písmeny jako druhý vstup, a to teď umíte.
- **Paleta.** Místo posterizace světlosti se barvy dají zaokrouhlit na paletu několika barev, vybranou ručně nebo shlukováním (k-means).

### Bonusové úkoly

1. **Barevné čáry.** Místo černé kreslete čáry barvou: tmavší verzí barvy plochy pod nimi, nebo pevnou barvou z výběru barvy (`source_color`).
2. **Tloušťka čáry.** Ztlušťte čáry XDoG tak, že vezmete minimum z okolí, a zjistěte, jak to vypadá s různým poloměrem.
3. **Půltóny.** Napište kernel, který plochu nahradí mřížkou černých teček s poloměrem podle světlosti, jako novinová fotka.
4. **Rychlá Kuwahara.** Součty ve čtvercích jdou spočítat z integrálního obrazu (*summed-area table*) v konstantním čase, nezávisle na poloměru. Napište to v NumPy přes `np.cumsum`.
""")

md(r"""
### Nápady na semestrální práci

- Plugin „komiks“: kreslený vzhled s půltónovými stíny, barevnými obrysy a volbou palety.
- Anizotropní Kuwaharův filtr s polem směrů ze Sobelova gradientu, jako malířský filtr.
- Generátor omalovánek: čáry XDoG, uzavřené plochy a jejich čísla podle barvy.
""")


def build():
    lesson_writer.write(cells, OUT, "l5")


if __name__ == "__main__":
    build()
