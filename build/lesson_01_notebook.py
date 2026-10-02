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
def threshold(ivec2 at) -> float:
    return (BAYER[(at.y % 4) * 4 + at.x % 4] + 0.5) / 16.0


def posterise(float t, int n, float d) -> float:
    float levels = float(n - 1)
    return floor(clamp(t, 0.0, 1.0) * levels + d) / levels"""

GOAL = f"""\
%%gmacs "balls" -> "Goal"
import linear_srgb_color_space
import oklab_color_space

{COLOURS}
uniform float mid_at: hint_range(0.05, 0.95) = 0.5
uniform int steps: hint_range(2, 16) = 5
uniform bool dither = true

{BAYER_GMACS}


{LAB}


{SOLUTION_GRADIENT}


{SOLUTION_DITHER}


def pixel(ivec2 at) -> vec4:
    vec4 c = src(at)
    float d = 0.5
    if dither:
        d = threshold(at)
    float t = posterise(to_lab(c.rgb).x, steps, d)
    return vec4(gradient(t), c.a)"""

HELPERS_NP = '''\
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
    "dither": True,
}'''

GRADIENT_NP_GIVEN = '''\
def gradient(t, dark, mid, light, mid_at):
    """Colours of shape (h, w, 3) for lightness t of shape (h, w)."""
    # TODO: dark to mid below mid_at, mid to light above it, mixed in OKLab
    return np.zeros(t.shape + (3,), dtype=np.float32) + np.asarray(dark, dtype=np.float32)'''

SOLUTION_GRADIENT_NP = '''\
def gradient(t, dark, mid, light, mid_at):
    """Colours of shape (h, w, 3) for lightness t of shape (h, w)."""
    t = t[..., None]                         # (h, w, 1) mixes with a colour of shape (3,)
    low = mix(to_lab(dark), to_lab(mid), t / mid_at)
    high = mix(to_lab(mid), to_lab(light), (t - mid_at) / (1.0 - mid_at))
    return from_lab(np.where(t < mid_at, low, high))'''

APPLY_NP_GIVEN = '''\
def apply(img, dark, mid, light, mid_at, steps, dither):
    out = img.copy()
    # TODO: lightness, threshold, posterise, gradient
    return out'''

SOLUTION_APPLY_NP = '''\
def apply(img, dark, mid, light, mid_at, steps, dither):
    out = img.copy()
    L = to_lab(img[..., :3])[..., 0]
    if dither:
        yy, xx = np.mgrid[0:L.shape[0], 0:L.shape[1]]
        d = (BAYER[yy % 4, xx % 4] + 0.5) / 16
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
# Lekce 1: Gradient map s ditheringem

Na konci lekce budete mít efekt, který obrázek přebarví podle světlosti přechodem tří barev, omezí ho na několik odstínů a mezery mezi nimi dorovná ditheringem. Cestou uvidíte, proč záleží na tom, v jakých číslech se barvy míchají, a proč se nejznámější dithering na GPU nehodí.

V Sáře otevřete nový dokument ve výchozí velikosti a spusťte buňku. Načte dva testovací obrázky jako vrstvy `balls` a `color_chart`.
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
## Cíl lekce

Buňka níže spustí hotový efekt na vrstvě `balls`. Její kód je schovaný, protože ho během lekce napíšete sami. Vyzkoušejte barvy, posuvník `steps` a přepínač `dither`.
""")

code(GOAL, hidden=True)

# --- 1. Mixing two colours ------------------------------------------------------
md(r"""
## 1. Míchání dvou barev

Gradient map přebarví pixel podle jeho světlosti $t$: černá dostane první barvu, bílá poslední a mezi nimi barvu z přechodu. Za světlost bereme $L$ z OKLab, číslo od 0 do 1. Se dvěma barvami se efektu říká duotone.

Přechod mezi barvami $a$ a $b$ je `mix(a, b, t)`. Jak vypadá, záleží na tom, v jakých číslech barvy smícháte:

@img(mix_spaces.png, 900, Tři přechody namíchané v sRGB, v lineárním RGB a v OKLab)

| Prostor | Co jsou čísla | K čemu |
|---|---|---|
| sRGB | hodnoty, jak je ukládá obrázek | k uložení, ne k počítání |
| lineární RGB | množství světla | kde se světlo sčítá: rozostření, zmenšení obrázku, průhlednost |
| OKLab | barva, jak ji vnímá oko | přechody a palety, které mají působit rovnoměrně |

Kernel níže je duotone na vrstvě `color_chart`. Přepínač `space` volí prostor, zatím funguje jen sRGB.

### 🎯 Úkol 1: míchání v lineárním RGB a v OKLab

Doplňte ve funkci `blend` větve pro lineární RGB (`space == 1`) a OKLab (`space == 2`). Funkce dostane barvy v sRGB a v sRGB musí výsledek i vrátit. Nedokončené větve vracejí barvu `a` na celém obrázku.

<details><summary>💡 Nápověda</summary>

1. Knihovny nabízejí čtyři převody: `srgb_to_linear_srgb`, `linear_srgb_to_srgb`, `linear_srgb_to_oklab` a `oklab_to_linear_srgb`. Do OKLab se jde přes lineární RGB.
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

Šedá škála dole na vrstvě je teď celý přechod od modré ke žluté. V OKLab má uprostřed světle modrozelenou barvu jako prostřední sloupec obrázku nahoře, v lineárním RGB světle šedou a v sRGB tmavší šedou.

> **❓ Otázka**
> Ve kterém prostoru zabírají tmavé modré odstíny na šedé škále nejkratší úsek a proč?

<details><summary>🔑 Odpověď</summary>

V lineárním RGB. Polovina přechodu je tam polovina světla a ta oku připadá světlá, šedá s polovinou světla má světlost asi 0.8. Tmavé odstíny se proto stlačí na začátek přechodu. OKLab dělí přechod podle vnímané světlosti, tmavá i světlá polovina jsou stejně dlouhé.
</details>
""")

# --- 2. A gradient map ----------------------------------------------------------
md(r"""
## 2. Přechod tří barev

Tři barvy dají stínům, středním tónům a světlům každému vlastní barvu. Posuvník `mid_at` určuje, při jaké světlosti leží prostřední barva:

@img(gradient_map.png, 900, Světlost pixelu se převede na barvu z přechodu tří barev)

### 🎯 Úkol 2: funkce `gradient`

Napište funkci `gradient(t)`. Pro `t` od 0 do `mid_at` míchá `dark` a `mid`, pro `t` od `mid_at` do 1 míchá `mid` a `light`, obojí v OKLab. Funkce `to_lab` a `from_lab` jsou převody z úkolu 1, zabalené do jedné funkce. Nedokončená buňka obarví celý obrázek barvou `dark`.

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

Posterizace zaokrouhlí světlost na $n$ úrovní, třeba pro paletu retro hry nebo kreslený vzhled. Pro $n = 5$ jsou úrovně 0, 0.25, 0.5, 0.75 a 1:

$$q = \frac{\lfloor t\,(n - 1) + d \rfloor}{n - 1}, \qquad d = 0.5$$

Přičtené $d = 0.5$ dělá z oříznutí desetinné části $\lfloor \cdot \rfloor$ zaokrouhlení na nejbližší úroveň. Na plynulém stínování míčků vzniknou ploché pruhy (*banding*).

**Dithering** místo pevného $d = 0.5$ přičte práh, který se mění od pixelu k pixelu. Sousední pixely se pak zaokrouhlí různě a v průměru zůstane plocha stejně světlá jako před zaokrouhlením. Uspořádaný dithering (*ordered dithering*) bere prahy z malé matice, která se opakuje přes celý obrázek. Bayerova matice 4 × 4 rozkládá prahy tak, aby se sousední pixely co nejvíc lišily:

@img(bayer.png, 900, Bayerova matice 4 × 4 a přechod převedený na dvě úrovně třemi způsoby)

### 🎯 Úkol 3: posterizace a Bayerův práh

Kernel ukazuje světlost míčků šedě. Napište dvě funkce:

- `posterise(t, n, d)` podle vzorce nahoře. Hodnota `t` mimo 0 až 1 nesmí dát úroveň mimo 0 až 1.
- `threshold(at)`, Bayerův práh pixelu na pozici `at`, číslo od 0 do 1 podle obrázku.

Pak zapněte `dither` a zkuste `steps` od 2 do 8. Nedokončená buňka vrací šedou bez zaokrouhlení.

<details><summary>💡 Nápověda</summary>

1. `n` je celé číslo. Než s ním začnete počítat ve `float`, převeďte ho: `float(n - 1)`.
2. Hodnotu do rozsahu omezí `clamp(x, lo, hi)`. Stačí omezit `t`, protože $d$ je menší než 1.
3. Matice se opakuje po čtyřech pixelech, sloupec v ní je `at.x % 4` a řádek `at.y % 4`.
4. `BAYER` je jedno pole o 16 číslech, řádek po řádku. Prvek v řádku `j` a sloupci `i` má index `j * 4 + i`.
</details>

<details><summary>🔑 Řešení</summary>

""" + answer(SOLUTION_DITHER) + r"""
</details>
""")

code(f"""
%%gmacs "balls" -> "Posterise"
import linear_srgb_color_space
import oklab_color_space

uniform int steps: hint_range(2, 16) = 4
uniform bool dither = false

{BAYER_GMACS}


def threshold(ivec2 at) -> float:
    # TODO: this pixel's Bayer threshold, from 0 to 1
    return 0.5


def posterise(float t, int n, float d) -> float:
    # TODO: round t to one of n levels from 0 to 1, d is the threshold
    return t


def pixel(ivec2 at) -> vec4:
    vec4 c = src(at)
    float d = 0.5
    if dither:
        d = threshold(at)
    float L = posterise(linear_srgb_to_oklab(srgb_to_linear_srgb(c.rgb)).x, steps, d)
    vec3 grey = linear_srgb_to_srgb(oklab_to_linear_srgb(vec3(L, 0.0, 0.0)))
    return vec4(grey, c.a)
""")

md(r"""
### ✅ Kontrola

Se `steps = 4` a vypnutým `dither` mají míčky čtyři ploché odstíny šedé. Po zapnutí `dither` se hranice mezi nimi rozpadnou na vzor teček a z dálky, nebo ve zmenšeném náhledu, vypadá stínování zase plynule.

### Proč ne Floyd-Steinberg

Nejznámější dithering, Floyd-Steinberg, prochází pixely po řádcích. Každý zaokrouhlí a chybu, o kterou se spletl, rozdělí sousedům, kteří ještě čekají:

@img(floyd_steinberg.png, 620, Chyba zaokrouhleného pixelu se rozdělí čtyřem sousedům s vahami 7, 3, 5 a 1 šestnáctin)

Výsledek nemá viditelný vzor a vypadá lépe než Bayer.

> **❓ Otázka**
> Proč se Floyd-Steinberg nehodí na GPU, kde má každý pixel vlastní vlákno? A proč se tam hodí Bayer?

<details><summary>🔑 Odpověď</summary>

Pixel může zaokrouhlit až po přičtení chyb od souseda vlevo a od tří sousedů nad sebou, a ti čekají na své sousedy. Výpočet je řetěz přes celý obrázek, vlákna by na sebe čekala jedno po druhém. Práh z Bayerovy matice závisí jen na pozici `(x, y)`, každé vlákno ho spočítá samo, nezávisle na ostatních.
</details>
""")

# --- 4. NumPy and Krita ---------------------------------------------------------
md(r"""
## 4. Do Krity

Teď celý efekt napíšete v NumPy. `PARAMS` popisuje stejné parametry jako kernel „Goal“: barvy jako `"#1b1035"`, z nichž udělá výběr barvy, a zaškrtávátko jako `True`. `apply` dostane barvy jako trojice čísel od 0 do 1 v sRGB.

Pomocné funkce a převody do OKLab už znáte, buňku stačí spustit. `to_lab` a `from_lab` převádějí jednu barvu i celé pole barev najednou.
""")

code(HELPERS_NP)

md(r"""
### 🎯 Úkol 4: přechod v NumPy

Napište `gradient(t, dark, mid, light, mid_at)` v NumPy. `t` je pole světlostí tvaru (výška, šířka), výsledek je pole barev tvaru (výška, šířka, 3). Nedokončená funkce vrací všude barvu `dark`.

<details><summary>💡 Nápověda</summary>

1. Podmínka `if` pro každý pixel se v NumPy píše jinak: spočítejte oba úseky pro všechny pixely a pro každý pixel vyberte jeden funkcí `np.where(podmínka, a, b)`.
2. `t` má tvar (výška, šířka), barva (3,). `t[..., None]` přidá osu a tvar (výška, šířka, 1) už jde s barvou násobit.
</details>

<details><summary>🔑 Řešení</summary>

""" + answer(SOLUTION_GRADIENT_NP) + r"""
</details>
""")

code(GRADIENT_NP_GIVEN)

md(r"""
### 🎯 Úkol 5: celý efekt

Napište `apply`: světlost $L$ obrázku, práh podle `dither`, posterizace na `steps` úrovní a nakonec barvy z `gradient`. Nedokončená funkce vrací obrázek beze změny.

<details><summary>💡 Nápověda</summary>

1. Světlost všech pixelů je `to_lab(img[..., :3])[..., 0]`.
2. Souřadnice všech pixelů dá `np.mgrid` jako v úvodní lekci. Pole jde indexovat polem: `BAYER[yy % 4, xx % 4]` je práh pro každý pixel najednou.
3. Bez ditheringu stačí `d = 0.5`. Číslo se s polem sečte stejně jako pole s polem.
4. `np.floor` a `np.clip` jsou `floor` a `clamp` z GLSL.
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

Porovnání s kernelem „Goal“ z úvodu. Kontrola počítá s výchozími hodnotami, pokud jste s jeho posuvníky hýbali, spusťte buňku „Goal“ znovu.

Posterizace zaokrouhluje, takže pixel na hranici dvou úrovní se na GPU a v NumPy může zaokrouhlit různě a rozdíl je pak celá úroveň. Podstatné je, kolik takových pixelů je: setiny procenta jsou v pořádku, desítky procent znamenají jinou matematiku.
""")

code("""
defaults = {"dark": (27 / 255, 16 / 255, 53 / 255), "mid": (194 / 255, 65 / 255, 45 / 255),
            "light": (1.0, 231 / 255, 163 / 255), "mid_at": 0.5, "steps": 5, "dither": True}
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
### 🎯 Úkol 6: gradient map v Kritě

Vložte efekt do `pga_filter/effect.py` a vyzkoušejte ho na vlastní fotce, i s výběrem. Okno filtru ukáže tři výběry barvy, dva posuvníky a zaškrtávátko.

<details><summary>💡 Nápověda</summary>

1. Do `effect.py` patří buňka s pomocnými funkcemi a `PARAMS` a vaše `gradient` a `apply`. Řádek `import numpy as np` nechte nahoře.
2. `TITLE = "Gradient map"` pojmenuje položku v menu. Okno filtru zavřete a znovu otevřete, Kritu restartovat nemusíte.
</details>

> **ℹ️ Poznámka**
> Efekt počítá s obrázkem v sRGB, jaký má běžná fotka. V dokumentu Krity s lineárním profilem, třeba v 32bitovém float, dostane `apply` lineární čísla a dekódování sRGB je navíc.
""")

# --- Summary ------------------------------------------------------------------
md(r"""
## Shrnutí

- Hodnoty sRGB slouží k uložení. Světlo se sčítá v lineárním RGB, přechody, které mají působit rovnoměrně, se míchají v OKLab.
- Gradient map převede světlost $L$ na barvu z přechodu. Podmínka vybere úsek a `t` se přepočítá na 0 až 1 uvnitř něj.
- Posterizace zaokrouhlí na $n$ úrovní. Dithering nahradí pevný práh 0.5 prahem z Bayerovy matice, který závisí jen na pozici pixelu, takže každé vlákno počítá samo.
- Floyd-Steinberg šíří chybu na sousedy a pixely na sebe musí čekat.
- V NumPy se podmínka pro každý pixel píše jako `np.where` nad celým polem a tvary polí se srovnají přidáním osy, `t[..., None]`.

### Co jsme vynechali

- **Barvy mimo gamut.** Přechod v OKLab může projít barvou, kterou sRGB neumí zobrazit, a ta se při zobrazení ořízne. Mapování gamutu přijde u HDR.
- **Jiné prahy.** Modrý šum (*blue noise*) je matice prahů bez viditelného vzoru, také počítatelná pro každý pixel zvlášť. Error diffusion umí na GPU jen po řádcích nebo po dlaždicích.
- **Přechod s libovolným počtem barev.** Potřebuje pole barev jako parametr, což kernel v buňce zatím neumí.

### Bonusové úkoly

1. **Bayer 8 × 8.** Větší matice dá jemnější vzor. Postaví se ze 4 × 4: čtyři kopie $4B$ s přičtenými 0, 2, 3 a 1 v rozích (levý horní, pravý horní, levý dolní, pravý dolní). Sestavte ji v NumPy a vyzkoušejte v `apply`.
2. **Retro paleta.** Posterizujte každý kanál R, G, B zvlášť na dvě úrovně, s ditheringem i bez. Kolik barev výsledek má? Porovnejte posterizaci hodnot sRGB a lineárních.
3. **Duha.** V OKLab jde barva zapsat i jako světlost, sytost $C$ a odstín $h$: $a = C \cos h$, $b = C \sin h$. Napište gradient map, kde světlost pixelu určí odstín.
""")

md(r"""
### Nápady na semestrální práci

- Gradient map s předvolbami palet. Rozbalovací seznam v `PARAMS` jako `["Sunset", "Ocean", "Game Boy"]` vybere sadu barev, výběry barvy ji doladí.
- Retro filtr s pevnou paletou, třeba čtyřmi zelenými odstíny Game Boye, a s ditheringem. Matice prahů může být parametr, v `PARAMS` jako `[[0, 2], [3, 1]]` se z ní stane mřížka čísel k úpravě.
""")

# --- Appendix: the API --------------------------------------------------------
md(r"""
## Příloha: přehled funkcí

Nové v této lekci. Ostatní funkce popisuje příloha úvodní lekce.

| Volání nebo zápis | Co dělá |
|---|---|
| `sara.live(apply, PARAMS, source="balls", target="...")` | ovládací prvky pro NumPy funkci, čte vrstvu `source` a zapisuje do vrstvy `target` |
| `sara.check("vrstva", pole)` | největší rozdíl mezi vrstvou a polem a podíl pixelů, které se liší |
| `uniform vec3 c: source_color = vec3(1.0, 0.5, 0.2)` | výběr barvy, kernel dostane čísla v sRGB |
| `uniform int i: hint_enum("A", "B") = 0` | rozbalovací seznam, kernel dostane pořadí volby od 0 |
| `uniform int n: hint_range(2, 16) = 4` | posuvník po celých číslech |
| `uniform bool b = false` | zaškrtávátko |
| `const float A[16] = float[16](...)` | pole konstant, jako v GLSL |
| `srgb_to_linear_srgb`, `linear_srgb_to_srgb` | dekódování a kódování sRGB, knihovna `linear_srgb_color_space` |
| `linear_srgb_to_oklab`, `oklab_to_linear_srgb` | převody mezi lineárním RGB a OKLab, knihovna `oklab_color_space` |
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
