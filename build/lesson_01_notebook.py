"""Builds the lesson 1 notebook in the lecture folder above this one.

The notebook is generated: edit this file and run it again, never the .ipynb
by hand, or the next build overwrites the edit. The pictures and the test
picture balls.png come from lesson_01_diagrams.py.

Markdown cells are raw strings, so LaTeX keeps single backslashes and braces.
A picture from imgs/1 is written as @img(file, width, alt text).

The solutions are constants near the top, so the markdown shows exactly the
code that test_lesson_01.py runs.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "lesson_01_gradient_map_dither.ipynb"
IMG = "imgs/1"
cells = []


def _pictures(text):
    return re.sub(r"@img\(([^,]+), (\d+), ([^)]*)\)",
                  lambda m: f'<img src="{IMG}/{m[1]}" width="{m[2]}" alt="{m[3]}">', text)


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": _pictures(text).strip("\n").splitlines(keepends=True)})


def code(text, hidden=False):
    metadata = {"jupyter": {"source_hidden": True}} if hidden else {}
    cells.append({"cell_type": "code", "metadata": metadata, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(keepends=True)})


# --- The kernels and the solutions ------------------------------------------------
# The colours are the hex colours of PARAMS, to four decimals.
COLOURS = """\
uniform vec3 dark: source_color = vec3(0.1059, 0.0627, 0.2078)
uniform vec3 mid: source_color = vec3(0.7608, 0.2549, 0.1765)
uniform vec3 light: source_color = vec3(1.0, 0.9059, 0.6392)"""

LAB = """\
def to_lab(vec3 srgb) -> vec3:
    return linear_srgb_to_oklab(srgb_to_linear_srgb(srgb))


def from_lab(vec3 lab) -> vec3:
    return linear_srgb_to_srgb(oklab_to_linear_srgb(lab))"""

BAYER_GMACS = """\
const float BAYER[16] = float[16](0.0, 8.0, 2.0, 10.0, 12.0, 4.0, 14.0, 6.0, 3.0, 11.0, 1.0, 9.0, 15.0, 7.0, 13.0, 5.0)"""

SOLUTION_BLEND = """\
if space == 1:
    vec3 lin = mix(srgb_to_linear_srgb(a), srgb_to_linear_srgb(b), t)
    return linear_srgb_to_srgb(lin)
if space == 2:
    vec3 lab = mix(linear_srgb_to_oklab(srgb_to_linear_srgb(a)), linear_srgb_to_oklab(srgb_to_linear_srgb(b)), t)
    return linear_srgb_to_srgb(oklab_to_linear_srgb(lab))"""

SOLUTION_GRADIENT = """\
def gradient(float t) -> vec3:
    if t < mid_at:
        return from_lab(mix(to_lab(dark), to_lab(mid), t / mid_at))
    return from_lab(mix(to_lab(mid), to_lab(light), (t - mid_at) / (1.0 - mid_at)))"""

SOLUTION_DITHER = """\
def bayer(ivec2 at) -> float:
    return (BAYER[(at.y % 4) * 4 + at.x % 4] + 0.5) / 16.0


def ign(ivec2 at) -> float:
    return fract(52.9829189 * fract(dot(vec2(at), vec2(0.06711056, 0.00583715))))


def posterise(float t, int n, float d) -> float:
    float levels = float(n - 1)
    return floor(clamp(t, 0.0, 1.0) * levels + d) / levels"""

# The labels are in the same order in both kernels and in PARAMS, whose first
# label is the template's default, so the goal kernel defaults to index 0.
DITHERS = '"Bayer", "White noise", "IGN", "Off"'

THRESHOLD = """\
def threshold(ivec2 at) -> float:
    if dither == 0:
        return bayer(at)
    if dither == 1:
        return sa_hash21(vec2(at))
    if dither == 2:
        return ign(at)
    return 0.5"""

GOAL = f"""\
%%gmacs "balls" -> "Goal"
import linear_srgb_color_space
import oklab_color_space
import noise

{COLOURS}
uniform float mid_at: hint_range(0.05, 0.95) = 0.5
uniform int steps: hint_range(2, 16) = 5
uniform int dither: hint_enum({DITHERS}) = 0

{BAYER_GMACS}


{LAB}


{SOLUTION_GRADIENT}


{SOLUTION_DITHER}


{THRESHOLD}


def pixel(ivec2 at) -> vec4:
    vec4 c = src(at)
    float t = posterise(to_lab(c.rgb).x, steps, threshold(at))
    return vec4(gradient(t), c.a)"""

HELPERS_NP = '''\
def fract(x):
    """Same as fract in GLSL: the part after the decimal point."""
    return x - np.floor(x)


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
    "light": "#ffe7a3",
    "mid_at": (0.5, 0.05, 0.95),
    "steps": (5, 2, 16),
    "dither": ["Bayer", "White noise", "IGN", "Off"],
}'''

GRADIENT_NP_GIVEN = '''\
def gradient(t, dark, mid, light, mid_at):
    """Colours of shape (h, w, 3) for lightness t of shape (h, w)."""
    # TODO: dark to mid below mid_at, mid to light above it, mixed in OKLab
    return np.zeros(t.shape + (3,), dtype=np.float32) + np.asarray(dark, dtype=np.float32)'''

SOLUTION_GRADIENT_NP = '''\
def gradient(t, dark, mid, light, mid_at):
    """Colours of shape (h, w, 3) for lightness t of shape (h, w)."""
    stops = to_lab([dark, mid, light])       # one OKLab colour per row
    lab = np.stack([np.interp(t, [0.0, mid_at, 1.0], stops[:, k]) for k in range(3)], axis=-1)
    return from_lab(lab)'''

APPLY_NP_GIVEN = '''\
def apply(img, dark, mid, light, mid_at, steps, dither):
    out = img.copy()
    # TODO: lightness, threshold, posterise, gradient
    return out'''

SOLUTION_APPLY_NP = '''\
def apply(img, dark, mid, light, mid_at, steps, dither):
    out = img.copy()
    L = to_lab(img[..., :3])[..., 0]
    yy, xx = np.mgrid[0:L.shape[0], 0:L.shape[1]]
    if dither == "Bayer":
        d = (BAYER[yy % 4, xx % 4] + 0.5) / 16
    elif dither == "White noise":
        d = hash21(xx, yy)
    elif dither == "IGN":
        d = fract(52.9829189 * fract(0.06711056 * xx + 0.00583715 * yy))
    else:
        d = 0.5
    levels = steps - 1
    t = np.floor(np.clip(L, 0.0, 1.0) * levels + d) / levels
    out[..., :3] = gradient(t, dark, mid, light, mid_at)
    return out'''


def answer(snippet, lang="python"):
    """A solution block's code fence."""
    return f"```{lang}\n{snippet}\n```"


# --- Title --------------------------------------------------------------------
md(r"""
# Lekce 1: Gradientní mapa s ditheringem

Na konci lekce budete mít efekt, který obrázek přebarví podle světlosti přechodem tří barev, omezí ho na několik odstínů a přechody mezi nimi vyhladí ditheringem. Cestou uvidíte, proč záleží na tom, v jakém prostoru se barvy míchají, jaký šum se k ditheringu hodí a proč se nejznámější dithering na GPU nehodí.

Spusťte buňku. Otevře v Sáře nový dokument v sRGB místo toho otevřeného a načte dva testovací obrázky jako vrstvy `balls` a `color_chart`.
""")

code("""
import sys
sys.path.insert(0, "lib")            # sara.py and its helpers live in lib/
import numpy as np
import sara

doc, layer, pixels = sara.init()
chart = doc.import_image("imgs/0/color_chart.png")   # a layer called "color_chart"
balls = doc.import_image("imgs/1/balls.png")         # a layer called "balls"
balls.show()
""")

md(r"""
### Nové funkce v této lekci

Funkce z úvodní lekce shrnuje její příloha. Sloupec Sekce říká, kde se funkce objeví poprvé.

**Kernel v buňce `%%gmacs`**

| Zápis | Co dělá | Sekce |
|---|---|---|
| `uniform vec3 c: source_color = vec3(1.0, 0.5, 0.2)` | výběr barvy, kernel dostane čísla v sRGB | 1 |
| `uniform int i: hint_enum("A", "B") = 0` | rozbalovací seznam, kernel dostane pořadí volby od 0 | 1 |
| `uniform int n: hint_range(2, 16) = 4` | posuvník po celých číslech | 3 |
| `const float A[16] = float[16](...)` | pole konstant, jako v GLSL | 3 |
| `floor(x)`, `fract(x)`, `clamp(x, 0.0, 1.0)` | zaokrouhlení dolů, desetinná část, omezení do intervalu | 3 |
| `at.x % 4` | zbytek po dělení celých čísel | 3 |
| `sa_hash21(vec2 p)` | bílý šum, číslo od 0 do 1 z pozice, knihovna `noise` | 3 |

**Sára v Pythonu**

| Volání | Co dělá | Sekce |
|---|---|---|
| `sara.live(apply, PARAMS, source="balls", target="...")` | `source` vybere vrstvu, ze které NumPy funkce čte | 4 |
| `"#1b1035"` a `["A", "B"]` v `PARAMS` | výběr barvy, `apply` dostane trojici čísel od 0 do 1 v sRGB, a rozbalovací seznam, `apply` dostane text volby | 4 |
| `sara.check("vrstva", pole, tolerance=0.01)` | vypíše i podíl pixelů, které se liší víc než o `tolerance` | 4 |

**NumPy**

| Volání | Co dělá | Sekce |
|---|---|---|
| `np.interp(t, xp, fp)` | po částech lineární funkce: v bodech `xp` má hodnoty `fp`, mezi nimi interpoluje | 4 |
| `np.floor(x)`, `np.clip(x, 0, 1)` | totéž co `floor` a `clamp` v kernelu, po prvcích | 4 |
| `BAYER[yy % 4, xx % 4]` | indexování polem indexů: každému pixelu vybere prvek matice 4 × 4 | 4 |
| `stops[:, k]` | sloupec `k` matice | 4 |
| `np.asarray(x, dtype=np.float32)` | seznam nebo pole převede na pole `float32` | 4 |
""")

md(r"""
## Cíl lekce

Buňka níže spustí hotový efekt na vrstvě `balls`. Její kód je schovaný, protože ho během lekce napíšete sami. Vyzkoušejte barvy, posuvník `steps` a seznam `dither`.
""")

code(GOAL, hidden=True)

# --- 1. Mixing two colours ------------------------------------------------------
md(r"""
## 1. Míchání dvou barev

Gradientní mapa (*gradient map*) přebarví pixel podle jeho světlosti $t$: černý pixel dostane první barvu, bílý poslední a ostatní barvu z přechodu mezi nimi. Za světlost bereme $L$ z OKLabu, číslo od 0 do 1. Gradientní mapa se dvěma barvami se nazývá duotone.

Přechod mezi barvami $a$ a $b$ je `mix(a, b, t)`. Jak vypadá, záleží na tom, v jakém prostoru barvy smícháte:

@img(mix_spaces.png, 900, Tři přechody namíchané v sRGB, v lineárním RGB a v OKLabu)

| Prostor | Co jsou čísla | K čemu |
|---|---|---|
| sRGB | hodnoty, jak je ukládá obrázek | k uložení, ne k počítání |
| lineární RGB | množství světla | ke sčítání světla: rozostření, zmenšení obrázku, průhlednost |
| OKLab | barva, jak ji vnímá oko | k přechodům a paletám, které mají působit rovnoměrně |

Kernel níže přebarví vrstvu `color_chart` dvěma barvami. Seznam `space` vybírá prostor, ve kterém se barvy míchají, zatím funguje jen sRGB.

### 🎯 Úkol 1: míchání v lineárním RGB a v OKLabu

Doplňte ve funkci `blend` větve pro lineární RGB (`space == 1`) a OKLab (`space == 2`). Funkce dostane barvy v sRGB a výsledek musí vrátit také v sRGB. Dokud větev chybí, je celý obrázek v barvě `a`.

<details><summary>💡 Nápověda</summary>

1. Knihovny nabízejí čtyři převody: `srgb_to_linear_srgb`, `linear_srgb_to_srgb`, `linear_srgb_to_oklab` a `oklab_to_linear_srgb`. Do OKLabu se převádí přes lineární RGB.
2. Převeďte obě krajní barvy, smíchejte je a výsledek převeďte zpátky. Převádí se barvy `a` a `b`, ne pixel.
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


def blend(vec3 a, vec3 b, float t) -> vec3:
    # a, b and the result are sRGB, as src() and the colour pickers give them
    if space == 1:
        return a                  # TODO: mix the amounts of light
    if space == 2:
        return a                  # TODO: mix in OKLab
    return mix(a, b, t)


def pixel(ivec2 at) -> vec4:
    vec4 c = src(at)
    float t = linear_srgb_to_oklab(srgb_to_linear_srgb(c.rgb)).x
    return vec4(blend(dark, light, t), c.a)
""")

md(r"""
### ✅ Kontrola

Šedá škála dole na vrstvě je teď přechod od modré ke žluté. V OKLabu je uprostřed světle modrozelený jako prostřední sloupec obrázku nahoře, v lineárním RGB světle šedý a v sRGB tmavší šedý.

> **❓ Otázka**
> Ve kterém prostoru zabírají tmavé modré odstíny na šedé škále nejkratší úsek a proč?

<details><summary>🔑 Odpověď</summary>

V lineárním RGB. V polovině přechodu je tam polovina světla a ta oku připadá světlá: šedá s polovinou světla má světlost asi 0.8. Tmavé odstíny se proto stlačí na začátek přechodu. OKLab dělí přechod podle vnímané světlosti, takže tmavá i světlá polovina jsou stejně dlouhé.
</details>
""")

# --- 2. A gradient map ----------------------------------------------------------
md(r"""
## 2. Přechod tří barev

Se třemi barvami dostanou stíny, střední tóny a světla každé svou barvu. Posuvník `mid_at` určuje, při jaké světlosti leží prostřední barva:

@img(gradient_map.png, 900, Světlost pixelu se převede na barvu z přechodu tří barev)

### 🎯 Úkol 2: funkce `gradient`

Napište funkci `gradient(t)`. Pro `t` od 0 do `mid_at` míchá `dark` a `mid`, pro `t` od `mid_at` do 1 míchá `mid` a `light`, obojí v OKLabu. `to_lab` a `from_lab` jsou převody z úkolu 1, každý zabalený do jedné funkce. Nedokončená buňka obarví celý obrázek barvou `dark`.

<details><summary>💡 Nápověda</summary>

1. V prvním úseku jde `t` od 0 do `mid_at`, ale `mix` potřebuje číslo od 0 do 1. Jak ho z `t` dostanete? Ve druhém úseku jde `t` od `mid_at` do 1.
2. Podmínka se píše `if t < mid_at:` s odsazeným blokem. `return` uvnitř bloku funkci ukončí, druhý úsek proto žádné `else` nepotřebuje.
</details>

<details><summary>🔑 Řešení</summary>

""" + answer(SOLUTION_GRADIENT) + r"""
</details>
""")

code(f"""
%%gmacs "color_chart" -> "Gradient map"
import linear_srgb_color_space
import oklab_color_space

{COLOURS}
uniform float mid_at: hint_range(0.05, 0.95) = 0.5


{LAB}


def gradient(float t) -> vec3:
    # TODO: dark to mid below mid_at, mid to light above it, mixed in OKLab
    return dark


def pixel(ivec2 at) -> vec4:
    vec4 c = src(at)
    return vec4(gradient(to_lab(c.rgb).x), c.a)
""")

md(r"""
### ✅ Kontrola

Šedá škála dole přechází od tmavě fialové přes červenou do světle žluté jako na obrázku a posuvník `mid_at` posouvá červenou doleva a doprava.
""")

# --- 3. Posterise and dither ----------------------------------------------------
md(r"""
## 3. Málo odstínů: posterizace a dithering

Posterizace zaokrouhlí světlost na $n$ úrovní. Hodí se pro paletu retro hry nebo pro kreslený vzhled. Pro $n = 5$ jsou úrovně 0, 0.25, 0.5, 0.75 a 1:

$$q = \frac{\lfloor t\,(n - 1) + d \rfloor}{n - 1}, \qquad d = 0.5$$

Přičtením $d = 0.5$ se z oříznutí desetinné části $\lfloor \cdot \rfloor$ stane zaokrouhlení na nejbližší úroveň. Na plynulém stínování míčků vzniknou ploché pruhy (*banding*).

**Dithering** místo pevného $d = 0.5$ přičte práh, který se mění od pixelu k pixelu. Sousední pixely se pak zaokrouhlí různě a v průměru zůstane plocha stejně světlá jako před zaokrouhlením. Uspořádaný dithering (*ordered dithering*) bere prahy z malé matice, která se opakuje přes celý obrázek. Bayerova matice 4 × 4 rozkládá prahy tak, aby se sousední pixely co nejvíc lišily:

@img(bayer.png, 900, Bayerova matice 4 × 4 a přechod převedený na dvě úrovně bez ditheringu a s Bayerovým prahem)

Práh ale nemusí pocházet z matice. Stačí číslo od 0 do 1, které se od pixelu k pixelu mění:

- **Bílý šum** (*white noise*) dá každému pixelu náhodné číslo, nezávislé na sousedech. V kernelu ho z pozice spočítá `sa_hash21(vec2(at))` z knihovny `noise`. Nemá žádný vzor, ale náhodná čísla tvoří shluky a obraz je zrnitý.
- **Interleaved gradient noise** (IGN) navrhl Jorge Jimenez pro hru Call of Duty: Advanced Warfare (2014). Je to jeden vzorec, $\operatorname{fract}$ je desetinná část čísla:

$$\mathrm{IGN}(x, y) = \operatorname{fract}\bigl(52.9829189 \cdot \operatorname{fract}(0.06711056\,x + 0.00583715\,y)\bigr)$$

Vnitřní $\operatorname{fract}$ je přechod, který se přes obrázek opakuje. Násobení velkým číslem ho rozseká na mnoho přechodů vložených do sebe (*interleaved gradients*). Sousední pixely tak dostanou hodně rozdílné prahy jako u Bayera, ale bez opakující se mřížky.

@img(noise.png, 900, Čtyři prahy a tmavá záře zaokrouhlená na osm úrovní s každým z nich)

Bayerův vzor je vidět a hodí se pro retro vzhled. Kde má dithering pruhy jen skrýt, poslouží lépe IGN.

> **👉 Tip**
> Acerola ve videu [Understanding The Graphics Of Silksong](https://www.youtube.com/watch?v=au9pce-xg5s) (anglicky, pruhy od 13. minuty) ukazuje pruhy v záři na tmavém pozadí, proč je Bayer neskryje a jak je IGN odstraní jedním řádkem shaderu.

### 🎯 Úkol 3: posterizace, Bayer a IGN

Kernel ukazuje světlost míčků šedě a seznam `dither` vybírá práh. Napište tři funkce:

- `posterise(t, n, d)` podle vzorce nahoře. Hodnota `t` mimo 0 až 1 nesmí dát úroveň mimo 0 až 1.
- `bayer(at)`, Bayerův práh pixelu na pozici `at`, číslo od 0 do 1 podle obrázku.
- `ign(at)` podle vzorce IGN.

Pak vyzkoušejte všechny volby `dither` se `steps` od 2 do 8. Nedokončená buňka vrací šedou bez zaokrouhlení.

<details><summary>💡 Nápověda</summary>

1. `n` je celé číslo. Než s ním začnete počítat ve `float`, převeďte ho: `float(n - 1)`.
2. Hodnotu do rozsahu omezí `clamp(x, lo, hi)`. Stačí omezit `t`, protože $d$ je menší než 1.
3. Matice se opakuje po čtyřech pixelech, sloupec v ní je `at.x % 4` a řádek `at.y % 4`.
4. `BAYER` je jedno pole o 16 číslech, řádek po řádku. Prvek v řádku `j` a sloupci `i` má index `j * 4 + i`.
5. Pro IGN převeďte pozici na desetinná čísla, `vec2(at)`. Součet $0.067\,x + 0.0058\,y$ je skalární součin dvou vektorů, v GLSL `dot`. Desetinnou část vrací `fract`.
</details>

<details><summary>🔑 Řešení</summary>

""" + answer(SOLUTION_DITHER) + r"""
</details>
""")

code(f"""
%%gmacs "balls" -> "Posterise"
import linear_srgb_color_space
import oklab_color_space
import noise

uniform int steps: hint_range(2, 16) = 4
uniform int dither: hint_enum({DITHERS}) = 3

{BAYER_GMACS}


def bayer(ivec2 at) -> float:
    # TODO: this pixel's Bayer threshold, from 0 to 1
    return 0.5


def ign(ivec2 at) -> float:
    # TODO: interleaved gradient noise at this pixel, from 0 to 1
    return 0.5


def posterise(float t, int n, float d) -> float:
    # TODO: round t to one of n levels from 0 to 1, d is the threshold
    return t


{THRESHOLD}


def pixel(ivec2 at) -> vec4:
    vec4 c = src(at)
    float L = posterise(linear_srgb_to_oklab(srgb_to_linear_srgb(c.rgb)).x, steps, threshold(at))
    vec3 grey = linear_srgb_to_srgb(oklab_to_linear_srgb(vec3(L, 0.0, 0.0)))
    return vec4(grey, c.a)
""")

md(r"""
### ✅ Kontrola

Se `steps = 4` a volbou `Off` mají míčky čtyři ploché odstíny šedé. Bayer rozpadne hranice mezi nimi na pravidelný vzor, bílý šum na zrno se shluky a IGN na jemný šum bez mřížky. Z dálky vypadá stínování ve všech třech případech zase plynule.

### Proč ne Floyd-Steinberg

Nejznámější dithering, Floyd-Steinberg, prochází pixely po řádcích. Každý pixel zaokrouhlí a chybu, kterou tím udělal, rozdělí sousedům, kteří přijdou na řadu později:

@img(floyd_steinberg.png, 620, Chyba zaokrouhleného pixelu se rozdělí čtyřem sousedům s vahami 7/16, 3/16, 5/16 a 1/16)

Výsledek nemá viditelný vzor a vypadá lépe než Bayer.

> **❓ Otázka**
> Proč se Floyd-Steinberg nehodí na GPU, kde má každý pixel vlastní vlákno? A proč se tam hodí Bayer?

<details><summary>🔑 Odpověď</summary>

Pixel se smí zaokrouhlit až po přičtení chyb od souseda vlevo a od tří sousedů nad ním, a ti zase čekají na své sousedy. Výpočet je řetěz přes celý obrázek, vlákna by na sebe čekala jedno po druhém. Bayer, bílý šum i IGN závisí jen na pozici `(x, y)`, takže si každé vlákno spočítá práh samo, nezávisle na ostatních.
</details>
""")

# --- 4. NumPy and Krita ---------------------------------------------------------
md(r"""
## 4. Do Krity

Teď celý efekt napíšete v NumPy. `PARAMS` popisuje stejné parametry jako kernel „Goal“: z barvy jako `"#1b1035"` se stane výběr barvy a ze seznamu textů rozbalovací seznam, jehož první volba je výchozí. `apply` dostane barvy jako trojice čísel od 0 do 1 v sRGB a ze seznamu text vybrané volby.

Pomocné funkce už znáte, nová je jen `hash21`, bílý šum stejný jako `sa_hash21` v kernelu. Buňku stačí spustit. `to_lab` a `from_lab` převádějí jednu barvu i celé pole barev najednou.
""")

code(HELPERS_NP)

md(r"""
### 🎯 Úkol 4: přechod v NumPy

Napište `gradient(t, dark, mid, light, mid_at)` v NumPy. `t` je pole světlostí tvaru (výška, šířka), výsledek je pole barev tvaru (výška, šířka, 3). Barvy se míchají v OKLabu jako v kernelu. Nedokončená funkce vrací všude barvu `dark`.

<details><summary>💡 Nápověda</summary>

1. Přechod je po částech lineární funkce, mezi každými dvěma sousedními barvami úsečka. NumPy ji spočítá pro celé pole najednou: `np.interp(t, xp, fp)`, kde `xp` jsou polohy barev a `fp` hodnoty v nich.
2. `np.interp` počítá s jedním číslem v každém bodě, barva v OKLabu má tři. Interpolujte každou složku zvlášť a složte je zpátky funkcí `np.stack(..., axis=-1)`.
3. `to_lab` převede všechny tři barvy najednou, když dostane jejich seznam.
</details>

<details><summary>🔑 Řešení</summary>

""" + answer(SOLUTION_GRADIENT_NP) + r"""
</details>
""")

code(GRADIENT_NP_GIVEN)

md(r"""
### 🎯 Úkol 5: celý efekt

Napište `apply`: světlost $L$ obrázku, práh podle `dither`, posterizace na `steps` úrovní a nakonec barvy z `gradient`. `dither` je text vybrané volby, třeba `"IGN"`. Nedokončená funkce vrací obrázek beze změny.

<details><summary>💡 Nápověda</summary>

1. Světlost všech pixelů je `to_lab(img[..., :3])[..., 0]`.
2. Souřadnice všech pixelů dá `np.mgrid` jako v úvodní lekci. Pole jde indexovat polem: `BAYER[yy % 4, xx % 4]` je práh pro každý pixel najednou.
3. Bílý šum je `hash21(xx, yy)`. IGN napíšete s `fract` stejně jako v kernelu, jen místo `dot` sečtete dva součiny.
4. Bez ditheringu stačí `d = 0.5`. Číslo se s polem sečte stejně jako pole s polem.
5. `np.floor` a `np.clip` jsou `floor` a `clamp` z GLSL.
</details>

<details><summary>🔑 Řešení</summary>

""" + answer(SOLUTION_APPLY_NP) + r"""
</details>
""")

code(APPLY_NP_GIVEN + """


sara.live(apply, PARAMS, source="balls", target="Goal NumPy")
""")

md(r"""
### ✅ Kontrola

Porovnání s kernelem „Goal“ z úvodu. Kontrola počítá s jeho výchozími hodnotami. Pokud jste s nimi hýbali, spusťte buňku „Goal“ znovu.

Posterizace zaokrouhluje, takže pixel na hranici dvou úrovní se na GPU a v NumPy může zaokrouhlit různě a rozdíl je pak celá úroveň. Podstatné je, kolik takových pixelů je: setiny procenta jsou v pořádku, desítky procent znamenají jinou matematiku.
""")

code("""
defaults = {"dark": (27 / 255, 16 / 255, 53 / 255), "mid": (194 / 255, 65 / 255, 45 / 255),
            "light": (1.0, 231 / 255, 163 / 255), "mid_at": 0.5, "steps": 5, "dither": "Bayer"}
sara.check("Goal", apply(balls.read(), **defaults))
""")

md(r"""
### Rychlost
""")

code("""
import time

img = balls.read()
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
### 🎯 Úkol 6: gradientní mapa v Kritě

Vložte efekt do `pga_filter/effect.py` a vyzkoušejte ho na vlastní fotce, také jen na vybrané oblasti. Okno filtru ukáže tři výběry barvy, dva posuvníky a rozbalovací seznam.

<details><summary>💡 Nápověda</summary>

1. Do `effect.py` patří buňka s pomocnými funkcemi a `PARAMS` a vaše `gradient` a `apply`. Řádek `import numpy as np` nechte nahoře.
2. `TITLE = "Gradient map"` pojmenuje položku v menu. Okno filtru zavřete a znovu otevřete, Kritu restartovat nemusíte.
</details>

> **ℹ️ Poznámka**
> Efekt počítá s obrázkem v sRGB, jaký má běžná fotka. V dokumentu Krity s lineárním profilem, obvykle u 32bitových desetinných čísel na kanál, dostane `apply` lineární čísla a dekódování sRGB tam nepatří.
""")

# --- Summary ------------------------------------------------------------------
md(r"""
## Shrnutí

- Hodnoty sRGB slouží k uložení. Světlo se sčítá v lineárním RGB, přechody, které mají působit rovnoměrně, se míchají v OKLabu.
- Gradientní mapa převede světlost $L$ na barvu z přechodu. V kernelu podmínka vybere úsek a `t` se přepočítá na 0 až 1 uvnitř něj, v NumPy to celé spočítá `np.interp`.
- Posterizace zaokrouhlí na $n$ úrovní. Dithering nahradí pevný práh 0.5 prahem, který se mění od pixelu k pixelu: Bayerova matice dá pravidelný vzor, bílý šum zrno se shluky a IGN šum bez vzoru. Všechny tři závisí jen na pozici pixelu, takže každé vlákno počítá samo.
- Floyd-Steinberg šíří chybu na sousedy a pixely na sebe musí čekat.
- V NumPy se výpočet pro každý pixel píše nad celým polem najednou: `np.interp` pro přechod, indexování polem pro Bayerovu matici.

### Co jsme vynechali

- **Barvy mimo gamut.** Přechod v OKLabu může projít barvou, kterou sRGB neumí zobrazit, a ta se při zobrazení ořízne. Mapování gamutu přijde v lekci o HDR.
- **Modrý šum a error diffusion.** Modrý šum (*blue noise*) jsou prahy bez vzoru i bez shluků, předem spočítané do malé textury, která se opakuje přes obrázek jako Bayerova matice. Error diffusion umí na GPU jen po řádcích nebo po dlaždicích.
- **Přechod s libovolným počtem barev.** V NumPy stačí delší seznamy pro `np.interp`. Kernel by potřeboval pole barev jako parametr, což buňka zatím neumí.

### Bonusové úkoly

1. **Bayer 8 × 8.** Větší matice dá jemnější vzor. Postaví se ze 4 × 4: čtyři kopie $4B$ s přičtenými 0, 2, 3 a 1 v rozích (levý horní, pravý horní, levý dolní, pravý dolní). Sestavte ji v NumPy a vyzkoušejte v `apply`.
2. **Retro paleta.** Posterizujte každý kanál R, G, B zvlášť na dvě úrovně, s ditheringem i bez. Kolik barev výsledek má? Porovnejte posterizaci hodnot sRGB a lineárních.
3. **Duha.** V OKLab jde barva zapsat i jako světlost, sytost $C$ a odstín $h$: $a = C \cos h$, $b = C \sin h$. Napište gradient map, kde světlost pixelu určí odstín.
""")

md(r"""
### Nápady na semestrální práci

- Gradientní mapa s předvolbami palet. Rozbalovací seznam v `PARAMS` jako `["Sunset", "Ocean", "Game Boy"]` vybere sadu barev, výběry barvy ji doladí.
- Retro filtr s pevnou paletou, třeba čtyřmi zelenými odstíny Game Boye, a s ditheringem. Matice prahů může být parametr, v `PARAMS` jako `[[0, 2], [3, 1]]` se z ní stane mřížka čísel k úpravě.
""")


def build():
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for i, cell in enumerate(cells):
        cell["id"] = f"l1-{i:02d}"
    OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(OUT, len(cells), "cells")


if __name__ == "__main__":
    build()
