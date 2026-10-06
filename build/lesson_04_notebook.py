"""Builds the lesson 4 notebook in the lecture folder above this one.

The notebook is generated: edit this file and run it again, never the .ipynb
by hand, or the next build overwrites the edit. The pictures and the blurred
test photo astronaut_blur.png come from lesson_04_diagrams.py. The sharp photo
astronaut.png and brick.png are scikit-image's sample pictures, in the public
domain and under CC0, as imgs/4/SOURCES.md says.

Markdown cells are raw strings, so LaTeX keeps single backslashes and braces.
A picture from imgs/4 is written as @img(file, width, alt text).

The solutions are constants near the top, so the markdown shows exactly the
code that test_lesson_04.py runs.
"""
import re
import textwrap
from pathlib import Path

import lesson_writer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "lesson_04_frequency_domain.ipynb"
IMG = "imgs/4"
SIDE = 512                    # the photos' side and the canvas's, the kernels' N
BLUR_SIGMA = 2.0              # the blur astronaut_blur.png was made with
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
WAVE = """\
%%gmacs "astronaut" -> "Wave"
uniform int u: hint_range(-32, 32) = 8
uniform int v: hint_range(-32, 32) = 4
uniform float phase: hint_range(0, 360) = 0.0

const float TAU = 6.28318530718


def pixel(ivec2 at) -> vec4:
    float t = TAU * float(u * at.x + v * at.y) / 512.0 + radians(phase)
    return vec4(vec3(0.5 + 0.5 * cos(t)), 1.0)"""

DFT_ROWS = f"""\
%%gmacs "Lightness" -> "DFT rows"
const int N = {SIDE}
const float TAU = 6.28318530718


# The DFT of each row: the pixel at (u, y) gets the coefficient of frequency u
# of row y, the real part in red and the imaginary part in green.
def pixel(ivec2 at) -> vec4:
    vec2 total = vec2(0.0)
    for x in range(0, N):
        # whole turns dropped while still an int, so cos and sin get a small angle
        float angle = -TAU * float((at.x * x) % N) / float(N)
        total += src(ivec2(x, at.y)).r * vec2(cos(angle), sin(angle))
    return vec4(total / float(N), 0.0, 1.0)"""

COLUMNS_GIVEN = """\
def pixel(ivec2 at) -> vec4:
    # TODO: the DFT of column at.x of the rows' DFT, whose values are complex
    return vec4(src(at).rg, 0.0, 1.0)"""

SOLUTION_COLUMNS = """\
def pixel(ivec2 at) -> vec4:
    vec2 total = vec2(0.0)
    for y in range(0, N):
        vec2 a = src(ivec2(at.x, y)).rg
        float angle = -TAU * float((at.y * y) % N) / float(N)
        vec2 w = vec2(cos(angle), sin(angle))
        total += vec2(a.x * w.x - a.y * w.y, a.x * w.y + a.y * w.x)
    return vec4(total / float(N), 0.0, 1.0)"""

DFT_COLUMNS = f"""\
%%gmacs "DFT rows" -> "DFT"
const int N = {SIDE}
const float TAU = 6.28318530718


{{columns}}"""

BLUR_GPU = f"""\
%%gmacs "astronaut" -> "Blur GPU"
import linear_srgb_color_space

uniform float sigma: hint_range(0.5, 8) = 3.0

const int N = {SIDE}


# A Gaussian blur in linear light, summed over a square of 4 sigma around the
# pixel. A neighbour past an edge is taken from the opposite edge.
def pixel(ivec2 at) -> vec4:
    int r = int(ceil(4.0 * sigma))
    vec3 total = vec3(0.0)
    float weight = 0.0
    for dy in range_inclusive(-r, r):
        for dx in range_inclusive(-r, r):
            float k = exp(-float(dx * dx + dy * dy) / (2.0 * sigma * sigma))
            ivec2 p = (at + ivec2(dx, dy) + N) % N
            total += k * srgb_to_linear_srgb(src(p).rgb)
            weight += k
    return vec4(linear_srgb_to_srgb(total / weight), 1.0)"""

# --- NumPy ------------------------------------------------------------------------
CONVERT_NP = '''\
def srgb_to_linear(c):
    a = np.abs(c)
    return np.sign(c) * np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    a = np.abs(c)
    return np.sign(c) * np.where(a <= 0.0031308, a * 12.92, 1.055 * a ** (1 / 2.4) - 0.055)'''

HELPERS_NP = CONVERT_NP + '''


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


def lightness(img):
    """OKLab L of an (h, w, 4) picture in sRGB, an (h, w) array."""
    lms = srgb_to_linear(img[..., :3]) @ M1.T
    return np.cbrt(lms) @ M2[0]


def rgba(a):
    """An opaque picture from an (h, w) grey or an (h, w, 3) colour array."""
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 2:
        a = np.dstack([a, a, a])
    return np.dstack([a, np.ones(a.shape[:2], np.float32)])'''

SPECTRUM_GIVEN = '''\
def spectrum(F):
    """The spectrum F as a grey picture: zero frequency in the middle, log magnitude, 0 to 1."""
    # TODO: shift, the logarithm of the magnitude, divided by its largest value
    return rgba(np.zeros(F.shape))'''

SOLUTION_SPECTRUM = '''\
def spectrum(F):
    """The spectrum F as a grey picture: zero frequency in the middle, log magnitude, 0 to 1."""
    m = np.log1p(np.abs(np.fft.fftshift(F)))
    return rgba(m / m.max())'''

WAVE_SPECTRUM = '''\
wave = doc.layer("Wave").read()
F = np.fft.fft2(wave[..., 0])
dots = np.argwhere(np.abs(F) > 1000)                # the coefficients that are not almost zero
print("dots (v, u):", ((dots + 256) % 512 - 256).tolist())    # index N - k is frequency -k
doc.new_layer("Wave spectrum", spectrum(F)).show()'''

GAUSSIAN_GIVEN = '''\
def gaussian_filter(h, w, sigma):
    """H of a Gaussian blur of sigma pixels, (h, w), its frequencies in np.fft.fft2's order."""
    # TODO: exp(-2 pi^2 sigma^2 (fx^2 + fy^2)) with the frequencies from np.fft.fftfreq
    return np.ones((h, w))'''

SOLUTION_GAUSSIAN = '''\
def gaussian_filter(h, w, sigma):
    """H of a Gaussian blur of sigma pixels, (h, w), its frequencies in np.fft.fft2's order."""
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    return np.exp(-2 * np.pi ** 2 * sigma ** 2 * (fx ** 2 + fy ** 2))'''

BLUR_GIVEN = '''\
def blur_fft(channel, sigma):
    """channel, an (h, w) array, blurred by a Gaussian of sigma pixels through its spectrum."""
    # TODO: the spectrum times H, transformed back
    return channel.copy()'''

SOLUTION_BLUR = '''\
def blur_fft(channel, sigma):
    """channel, an (h, w) array, blurred by a Gaussian of sigma pixels through its spectrum."""
    H = gaussian_filter(*channel.shape, sigma)
    return np.fft.ifft2(np.fft.fft2(channel) * H).real'''

WIENER_GIVEN = '''\
def wiener(channel, sigma, K):
    """channel without the Gaussian blur of sigma pixels, K the noise's power to the signal's."""
    # TODO: the spectrum times conj(H) / (|H|^2 + K), transformed back
    return channel.copy()'''

SOLUTION_WIENER = '''\
def wiener(channel, sigma, K):
    """channel without the Gaussian blur of sigma pixels, K the noise's power to the signal's."""
    G = np.fft.fft2(channel)
    H = gaussian_filter(*channel.shape, sigma)
    F = G * np.conj(H) / (np.abs(H) ** 2 + K)
    return np.fft.ifft2(F).real'''

PARAMS_NP = f'''\
PARAMS = {{
    "mode": ["Deblur", "Low pass", "High pass"],
    "sigma": ({BLUR_SIGMA}, 0.5, 8.0),
    "noise": (0.01, 0.0, 0.1, 0.001),
}}'''

APPLY_GIVEN = '''\
def apply(img, mode, sigma, noise):
    out = img.copy()
    # TODO: the three modes in linear RGB, High pass from the sRGB values
    return out'''

SOLUTION_APPLY = '''\
def apply(img, mode, sigma, noise):
    out = img.copy()
    lin = srgb_to_linear(img[..., :3])
    for c in range(3):
        if mode == "Deblur":
            lin[..., c] = wiener(lin[..., c], sigma, noise ** 2)
        else:
            lin[..., c] = blur_fft(lin[..., c], sigma)
    out[..., :3] = linear_to_srgb(lin)
    if mode == "High pass":
        out[..., :3] = img[..., :3] - out[..., :3] + 0.5
    return out'''

GOAL = ("# The filter of section 5, hidden because you write it yourself.\n"
        "def _goal():\n"
        + textwrap.indent("\n\n".join(piece.replace("\n\n\n", "\n\n") for piece in
                                        [CONVERT_NP, SOLUTION_GAUSSIAN, SOLUTION_BLUR, SOLUTION_WIENER, SOLUTION_APPLY]), "    ")
        + "\n\n    return apply\n\n\n"
        + PARAMS_NP
        + '\n\nsara.live(_goal(), PARAMS, source="astronaut_blur", target="Goal")')

DEFAULTS = f'{{"mode": "Deblur", "sigma": {BLUR_SIGMA}, "noise": 0.01}}'


def answer(snippet, lang="python"):
    """A solution block's code fence."""
    return f"```{lang}\n{snippet}\n```"


# --- Title --------------------------------------------------------------------
md(r"""
# Lekce 4: Frekvenční oblast

Na konci lekce budete mít filtr, který rozmazanou fotku zaostří. Rozmazání se nedá vrátit pixel po pixelu, ve spektru obrázku je to ale obyčejné násobení, a to se vrátit dá. Cestou spočítáte Fourierovu transformaci na GPU i v NumPy, naučíte se číst spektrum a rozmažete fotku přes frekvence.

Spusťte buňku. Otevře v Sáře nový dokument v sRGB s plátnem 512 × 512, velikostí obou fotek, a fotky otevře jako vrstvy. Fotka `astronaut_blur.png` je ostrá fotka `astronaut.png` rozmazaná Gaussem.
""")

code("""
import sys
sys.path.insert(0, "lib")            # sara.py and its helpers live in lib/
import time
import numpy as np
import sara

doc, layer, pixels = sara.init(width=512, height=512)   # a canvas the size of the photos
photo = doc.import_image("imgs/4/astronaut.png")        # a layer called "astronaut"
blurry = doc.import_image("imgs/4/astronaut_blur.png")  # a layer called "astronaut_blur"
""")

md(r"""
### Nové funkce v této lekci

Funkce z úvodní lekce shrnuje její příloha. Sloupec Sekce říká, kde se funkce objeví poprvé.

**Sára v Pythonu**

| Volání | Co dělá | Sekce |
|---|---|---|
| `sara.init(width=512, height=512)` | jako `sara.init()`, jen s plátnem dané šířky a výšky | úvod |

**Kernel v buňce `%%gmacs`**

| Zápis | Co dělá | Sekce |
|---|---|---|
| `const int N = 512` | celočíselná konstanta | 2 |
| `a % b` | zbytek po dělení celých čísel, u `ivec2` po složkách | 2, 3 |
| `vec2` jako komplexní číslo | reálná část v `x`, imaginární v `y` | 2 |
| `exp(x)`, `ceil(x)` | $e^x$ a zaokrouhlení nahoru | 3 |

**NumPy**

| Volání | Co dělá | Sekce |
|---|---|---|
| `np.fft.fft2(a)`, `np.fft.ifft2(F)` | dvourozměrná Fourierova transformace a zpětná transformace | 1, 3 |
| `np.fft.fftshift(F)` | přesune nulovou frekvenci doprostřed pole | 1 |
| `np.abs(F)`, `F.real`, `np.conj(F)` | absolutní hodnota, reálná část a komplexně sdružené číslo, po prvcích | 1, 3, 4 |
| `np.log1p(x)` | $\ln(1 + x)$ po prvcích | 1 |
| `1j` | imaginární jednotka | 2 |
| `np.fft.fftfreq(n)` | frekvence `n` koeficientů v pořadí, v jakém je vrací `np.fft.fft2` | 3 |
""")

md(r"""
## Cíl lekce

Buňka níže spustí hotový filtr na rozmazanou fotku. Jeho kód je schovaný, protože ho během lekce napíšete sami. Režim `Deblur` fotku zaostří, `Low pass` rozmaže a `High pass` nechá jen detaily. Vyzkoušejte, co dělá posuvník `noise` u režimu `Deblur`, hlavně blízko nuly.
""")

code(GOAL, hidden=True)

# --- 1. The spectrum ----------------------------------------------------------
md(r"""
## 1. Spektrum

Každý obrázek se dá složit ze **základních vln**, obrázků s průběhem kosinu. Vlna má frekvenci $(u, v)$: na šířku plátna se do ní vejde $u$ period a na výšku $v$ period. Vektor $(u, v)$ míří ve směru, ve kterém se vlna mění, a jeho délka říká, jak hustě. **Fourierova transformace** pro každou frekvenci zjistí, kolik té vlny obrázek obsahuje a jak je posunutá. Pro obrázek $f$ o straně $N$ pixelů:

$$F(u, v) = \sum_{x=0}^{N-1} \sum_{y=0}^{N-1} f(x, y)\, e^{-2\pi i\,(ux + vy)/N}$$

Podle Eulerova vzorce je $e^{-i\varphi} = \cos\varphi - i \sin\varphi$, takže suma násobí obrázek kosinem a sinem frekvence $(u, v)$. Výsledek $F(u, v)$ je komplexní číslo: jeho absolutní hodnota $|F(u, v)|$ říká, jak silná vlna v obrázku je, jeho úhel (fáze) říká, kam je posunutá. Obrázek velikostí $|F|$ pro všechny frekvence je **spektrum**.

@img(waves.png, 900, Čtyři základní vlny s různými frekvencemi u a v a pod každou její spektrum, dva body souměrné kolem středu a bod uprostřed)

Kernel níže nakreslí jednu vlnu.
""")

code(WAVE)

md(r"""
Pomocné funkce obsahují převody z lekce o barevných prostorech, `lightness`, která z obrázku spočítá světlost OKLab L, a `rgba`, která z šedého nebo barevného pole udělá neprůhledný obrázek. Buňku stačí spustit.
""")

code(HELPERS_NP)

md(r"""
NumPy spočítá transformaci funkcí `np.fft.fft2` algoritmem FFT (*fast Fourier transform*). Vrátí pole komplexních čísel stejného tvaru jako obrázek, ale v nezvyklém pořadí: nulová frekvence je v levém horním rohu a záporné frekvence jsou na konci pole, index $N - 1$ je frekvence $-1$. `np.fft.fftshift` prohodí kvadranty, takže nulová frekvence je uprostřed a frekvence rostou od středu ke krajům:

@img(fftshift.png, 900, Pořadí koeficientů z np.fft.fft2 s nulovou frekvencí v rohu a po np.fft.fftshift s nulovou frekvencí uprostřed, na schématu a na spektru fotky)

Velikosti se liší o mnoho řádů. Koeficient nulové frekvence je součet všech pixelů, koeficienty jemných detailů bývají tisíckrát menší. Spektrum se proto kreslí v logaritmu, $\ln(1 + |F|)$, a jednička uvnitř drží nulu na nule.

### 🎯 Úkol 1: spektrum

Napište `spectrum(F)`, která z pole koeficientů `F` udělá šedý obrázek: nulová frekvence uprostřed, logaritmus velikosti, vydělený svým maximem, aby byl od 0 do 1. Buňka pod funkcí ukáže spektrum vlny. Vlna je šedá, stačí z ní vzít červený kanál. Nedokončená funkce vrací černou.

<details><summary>💡 Nápověda</summary>

1. Pořadí kroků: přesunout, velikost, logaritmus, dělení maximem.
2. Velikost komplexního čísla dává `np.abs`, logaritmus z jedničky plus čísla `np.log1p`.
3. Šedé pole na obrázek převede `rgba`.
</details>

<details><summary>🔑 Řešení</summary>

""" + answer(SOLUTION_SPECTRUM) + r"""
</details>
""")

code(SPECTRUM_GIVEN + "\n\n\n" + WAVE_SPECTRUM, solve=[(SPECTRUM_GIVEN, SOLUTION_SPECTRUM)])

md(r"""
### ✅ Kontrola

Spektrum vlny jsou tři světlé pixely, v Sáře si střed vrstvy `Wave spectrum` přibližte. Buňka vypíše jejich frekvence jako $(v, u)$. Prostřední pixel je nulová frekvence, průměrný jas vlny. Dva další leží souměrně kolem středu, $u$ pixelů vodorovně a $v$ pixelů svisle od něj. Posuňte posuvníky `u` a `v`, spusťte buňku se spektrem znovu a body se posunou. Posuvník `phase` vlnu posune, ale spektrum nezmění.

> **❓ Otázka**
> Vlna má jednu frekvenci $(u, v)$. Proč má spektrum dva body, druhý v $(-u, -v)$?

<details><summary>🔑 Odpověď</summary>

Kosinus je součet dvou komplexních vln s opačnou frekvencí, $\cos\varphi = \frac{1}{2}\left(e^{i\varphi} + e^{-i\varphi}\right)$. Totéž platí pro každý obrázek s reálnými čísly: koeficient $F(-u, -v)$ je komplexně sdružený s $F(u, v)$, má stejnou velikost a opačnou fázi. Spektrum reálného obrázku je proto vždy souměrné podle středu a jeho druhá polovina nenese nic nového. Fázi `phase` nese úhel koeficientů, ne jejich velikost, proto se spektrum s ní nemění.
</details>

Teď spektrum fotky, ze světlosti:
""")

code("""
L = lightness(photo.read())
doc.new_layer("Spectrum", spectrum(np.fft.fft2(L))).show()
""")

md(r"""
Jak spektrum číst:

- Blízko středu jsou velké útvary, daleko od středu jemné detaily. Spektrum fotky od středu rychle slábne, protože velké plochy nesou ve fotce víc energie než detaily.
- Hrana v obrázku dá ve spektru čáru kolmo na hranu. Otočení obrázku otočí i spektrum.
- Co je v obrázku úzké, je ve spektru široké, a naopak. Obdélník dá funkci $\operatorname{sinc}$ s vlnkami, Gauss dá opět Gauss.
- Vzor, který se opakuje, dá mřížku jasných bodů. Čím delší je perioda vzoru, tím blíž jsou body u sebe.
- Posunutí obrázku změní jen fázi, spektrum velikostí zůstane stejné.

@img(pairs.png, 900, Obrázky nahoře a jejich spektra dole: úzký a široký obdélník, Gauss, otočený obdélník, opakovaný vzor a fotka)

> **❓ Otázka**
> Spektrum fotky má přes střed světlou vodorovnou a svislou čáru, i když fotka žádné takové hrany nemá. Odkud se vzaly?

<details><summary>🔑 Odpověď</summary>

Transformace počítá s obrázkem, jako by se opakoval dokola jako dlaždice: za pravým okrajem pokračuje levý, pod dolním horní. Okraje fotky na sebe nenavazují, a kde se potkají, vznikne skok, svislá hrana po celé výšce a vodorovná po celé šířce. Ty dají ve spektru čáry kolmo na ně. Zmizí, když se fotka před transformací vynásobí oknem (*window*), které k okrajům plynule klesá k nule.
</details>
""")

# --- 2. The DFT in a kernel ---------------------------------------------------
md(r"""
## 2. Fourierova transformace v kernelu

`np.fft.fft2` počítá na CPU. Kernel na GPU může spočítat přímo vzorec ze sekce 1, pixel výstupu $(u, v)$ je jedno vlákno. Dvojitá suma se rozdělí na dvě jednoduché, nejdřív přes $x$ v každém řádku, pak přes $y$ v každém sloupci výsledku:

$$G(u, y) = \frac{1}{N}\sum_{x=0}^{N-1} f(x, y)\, e^{-2\pi i\, ux/N}, \qquad F(u, v) = \frac{1}{N}\sum_{y=0}^{N-1} G(u, y)\, e^{-2\pi i\, vy/N}$$

Každý průchod sčítá $N$ hodnot na pixel místo $N^2$. Dělení $N$ v obou průchodech drží čísla malá, výsledek je `np.fft.fft2` dělené $N^2$.

Komplexní číslo uloží kernel do `vec2`, reálnou část do `x` a imaginární do `y`, a vrstva ho nese v červeném a zeleném kanálu. Vlna $e^{-i\varphi}$ je v kernelu `vec2(cos(angle), sin(angle))` s úhlem `angle` rovným $-\varphi$. Kernel transformuje světlost, kterou do vrstvy `Lightness` zapíše buňka níže:
""")

code("""
doc.new_layer("Lightness", rgba(L))
""")

md(r"""
První průchod, přes řádky, je hotový. Vstup je reálný, takže stačí násobit číslem:
""")

code(DFT_ROWS)

md(r"""
### 🎯 Úkol 2: transformace sloupců

Napište druhý průchod, přes sloupce vrstvy `DFT rows`. Vstup je tu už komplexní, a dvě komplexní čísla se násobí podle

$$(a + bi)(c + di) = (ac - bd) + (ad + bc)\,i$$

Nedokončený kernel výsledek prvního průchodu jen opíše, takže spektrum v kontrole bude transformace jen ve směru řádků.

<details><summary>💡 Nápověda</summary>

1. Cyklus je stejný jako v prvním průchodu, jen jde přes `y` a frekvence je `at.y`.
2. Hodnotu prvního průchodu přečte `src(ivec2(at.x, y)).rg`.
3. Vlnu si uložte do `vec2 w` a násobení podle vzorce napište po složkách: výsledek je `vec2(reálná část, imaginární část)`.
</details>

<details><summary>🔑 Řešení</summary>

""" + answer(SOLUTION_COLUMNS) + r"""
</details>
""")

code(DFT_COLUMNS.format(columns=COLUMNS_GIVEN), solve=[(COLUMNS_GIVEN, SOLUTION_COLUMNS)])

md(r"""
### ✅ Kontrola

Vrstva `DFT` je skoro celá černá, jen v levém horním rohu svítí bod. Je to nulová frekvence, průměrná světlost fotky, a ostatní koeficienty jsou tisíckrát menší. Buňka níže porovná vrstvu s `np.fft.fft2` a ukáže spektrum spočítané na GPU. Mělo by vypadat stejně jako spektrum fotky výše.
""")

code("""
dft = doc.layer("DFT").read()
F_gpu = (dft[..., 0] + 1j * dft[..., 1]) * 512 ** 2       # back to the scale of np.fft.fft2
F = np.fft.fft2(L)
sara.check("DFT", np.dstack([F.real, F.imag, np.zeros_like(L)]) / 512 ** 2, tolerance=1e-4)
doc.new_layer("Spectrum GPU", spectrum(F_gpu)).show()
""")

md(r"""
### Rychlost

Buňka změří FFT v NumPy a oba průchody na GPU. Kernel by v hotovém filtru četl a zapisoval vrstvy na GPU, NumPy musí vrstvu nejdřív přečíst a výsledek zapsat zpátky. Proto buňka měří i čtení a zápis vrstvy, zvlášť.
""")

code("""
times = []
for _ in range(5):
    start = time.perf_counter()
    np.fft.fft2(L)
    times.append(time.perf_counter() - start)

rows = sara.bench("DFT rows", runs=10)
columns = sara.bench("DFT", runs=10)

start = time.perf_counter()
photo.read()
read_ms = (time.perf_counter() - start) * 1000
start = time.perf_counter()
doc.layer("Lightness").write(rgba(L))
write_ms = (time.perf_counter() - start) * 1000

print(f"NumPy fft2:          {min(times) * 1000:8.2f} ms")
print(f"GPU, both passes:    {rows.gpu_ms + columns.gpu_ms:8.2f} ms")
print(f"reading a layer:     {read_ms:8.2f} ms")
print(f"writing a layer:     {write_ms:8.2f} ms")
""")

md(r"""
> **❓ Otázka**
> Kolikrát déle by oba způsoby počítaly obrázek 4096 × 4096, v každém směru osmkrát větší? Který byste pro něj zvolili?

<details><summary>🔑 Odpověď</summary>

Kernel sčítá v každém průchodu $N$ hodnot pro každý z $N^2$ pixelů, práce roste s $N^3$ a osmkrát větší strana je $8^3 = 512$krát víc práce. FFT potřebuje řádově $N^2 \log_2 N$ operací, tedy $64 \cdot 12 / 9 \approx 85$krát víc. Čtení a zápis vrstvy rostou s počtem pixelů, 64krát. Pro velký obrázek proto vyhraje FFT i s přenosem. Kernel se vyplatí tam, kde není potřeba celá transformace: pár vybraných frekvencí, nebo malé okolí pixelu jako u konvoluce v další sekci. Algoritmus FFT jde napsat i na GPU, jako $\log_2 N$ průchodů za sebou, a knihovny jako VkFFT nebo cuFFT to dělají. Buňka `%%gmacs` ale spouští jeden kernel.
</details>
""")

# --- 3. Blur in frequency -----------------------------------------------------
md(r"""
## 3. Rozmazání ve frekvencích

**Věta o konvoluci**: konvoluce obrázku s jádrem je ve spektru násobení. Spektrum rozmazaného obrázku je spektrum obrázku krát spektrum jádra, koeficient po koeficientu:

$$\mathcal{F}\{f * g\} = \mathcal{F}\{f\} \cdot \mathcal{F}\{g\}$$

Spektrum Gaussova jádra se směrodatnou odchylkou $\sigma$ pixelů je opět Gauss, tím užší, čím je jádro širší:

$$H(f_x, f_y) = e^{-2\pi^2 \sigma^2 (f_x^2 + f_y^2)}$$

$f_x$ a $f_y$ jsou frekvence v periodách na pixel, $u / N$ a $v / N$, od $-0.5$ do $0.5$. $H$ je u nulové frekvence 1 a k vysokým frekvencím klesá k nule. Rozmazání tedy ztlumí jemné detaily a velké plochy nechá, je to **dolní propust** (*low pass*).

@img(convolution.png, 900, Nahoře konvoluce fotky s Gaussovým jádrem, dole táž operace ve spektru jako násobení spektra fotky spektrem jádra)

Kernel níže rozmaže fotku přímo, váženým součtem okolí v lineárním RGB, jako v lekci o konvoluci. Tam jste rozmazání rozdělili na dva průchody, tady je pro jednoduchost v jednom. Soused za okrajem plátna se bere z protějšího okraje, stejně jako u transformace.
""")

code(BLUR_GPU)

md(r"""
### 🎯 Úkol 3: Gauss ve frekvencích

Napište `gaussian_filter(h, w, sigma)`, pole $H$ tvaru (h, w) s frekvencemi ve stejném pořadí, v jakém je vrací `np.fft.fft2`, a `blur_fft(channel, sigma)`, která kanál rozmaže přes spektrum. Nedokončené funkce kanál nemění.

<details><summary>💡 Nápověda</summary>

1. `np.fft.fftfreq(n)` vrátí frekvence `n` koeficientů v periodách na pixel, ve stejném pořadí jako `np.fft.fft2`.
2. Frekvence řádků udělejte sloupcem tvaru (h, 1) a frekvence sloupců řádkem tvaru (1, w), jako polohy v lekci o šumu. NumPy z nich spočítá celé pole.
3. `blur_fft`: transformace, násobení $H$, zpětná transformace `np.fft.ifft2`. Výsledek je komplexní s imaginární částí skoro nulovou, vraťte `.real`.
</details>

<details><summary>🔑 Řešení</summary>

""" + answer(SOLUTION_GAUSSIAN + "\n\n\n" + SOLUTION_BLUR) + r"""
</details>
""")

code(GAUSSIAN_GIVEN + "\n\n\n" + BLUR_GIVEN, solve=[(GAUSSIAN_GIVEN, SOLUTION_GAUSSIAN), (BLUR_GIVEN, SOLUTION_BLUR)])

md(r"""
### ✅ Kontrola

Buňka rozmaže fotku přes spektrum se stejnou `sigma`, jakou má kernel ve výchozím stavu, a porovná oba výsledky. Pokud jste posuvníkem kernelu hýbali, nastavte `sigma` v buňce na jeho hodnotu.
""")

code("""
sigma = 3.0                                    # the kernel's sigma
lin = srgb_to_linear(photo.read()[..., :3])
blurred = rgba(linear_to_srgb(np.dstack([blur_fft(lin[..., c], sigma) for c in range(3)])))
sara.check("Blur GPU", blurred)
doc.new_layer("Blur FFT", blurred).show()
""")

md(r"""
Kernel a FFT se liší nanejvýš o tisíciny. Kernel sčítá okolí do vzdálenosti $4\sigma$ a vzorkuje Gauss po pixelech, $H$ je spektrum Gaussovy křivky bez useknutí.
""")

code("""
times = []
for _ in range(5):
    start = time.perf_counter()
    for c in range(3):
        blur_fft(lin[..., c], sigma)
    times.append(time.perf_counter() - start)

t = sara.bench("Blur GPU", runs=10)
print(f"NumPy FFT:  {min(times) * 1000:8.2f} ms")
print(f"GPU kernel: {t.gpu_ms:8.2f} ms")
""")

md(r"""
> **❓ Otázka**
> Nastavte posuvník `sigma` kernelu na 8, v buňce s kontrolou `sigma = 8.0`, a spusťte kontrolu i měření znovu. Proč čas kernelu vzrostl a čas FFT ne? A proč horní okraj rozmazané fotky ztmavl a zčervenal?

<details><summary>🔑 Odpověď</summary>

Kernel sčítá čtverec o straně $2 \cdot 4\sigma + 1$ pixelů, takže jeho čas roste s $\sigma^2$, rozdělený na dva průchody s $\sigma$. FFT násobí každý koeficient číslem $H$ jednou, ať je $\sigma$ jakákoli. Malé rozmazání obrázku, který už je na GPU, je proto práce pro kernel, velké rozmazání pro FFT.

Transformace bere obrázek jako dlaždici, která se opakuje dokola, a násobení spektra je proto konvoluce dokola: horní okraj se rozmaže s dolním, kde je oranžový skafandr a černá přilba. Kernel tu dělá totéž, aby výsledky šly porovnat. Bez toho by se obrázek před transformací rozšířil o odraz svých okrajů (`np.pad`) a výsledek se pak ořízl.
</details>
""")

# --- 4. Wiener deconvolution --------------------------------------------------
md(r"""
## 4. Wienerova dekonvoluce

Když je rozmazání násobení spektra číslem $H$, mělo by jít vrátit dělením. Fotka `astronaut_blur` je fotka `astronaut` rozmazaná Gaussem se $\sigma = 2$ v lineárním RGB, s okraji dokola, a uložená v osmi bitech na kanál. Buňka její spektrum vydělí $H$ z úkolu 3:
""")

code(f"""
lin_blurry = srgb_to_linear(blurry.read()[..., :3])
H = gaussian_filter(512, 512, {BLUR_SIGMA})
divided = np.dstack([np.fft.ifft2(np.fft.fft2(lin_blurry[..., c]) / H).real for c in range(3)])
doc.new_layer("Inverse filter", rgba(linear_to_srgb(divided))).show()
""")

md(r"""
Vyšel šum. Uložení v osmi bitech zaokrouhlilo každý pixel až o 1/510, a to je šum na všech frekvencích stejně silný. U vysokých frekvencí je $H$ tak malé, že ve spektru rozmazané fotky zbyl skoro jen ten šum, a dělení ho zvětší tolikrát, kolikrát je $H$ menší než 1. **Wienerův filtr** dělí jen tam, kde je signál silnější než šum:

$$\hat F = G \cdot \frac{\overline{H}}{|H|^2 + K}$$

$G$ je spektrum rozmazané fotky, $\overline{H}$ komplexně sdružené $H$ a $K$ poměr výkonu šumu k výkonu signálu. Kde je $|H|^2$ mnohem větší než $K$, je zlomek skoro $1 / H$. Kde je mnohem menší, je zlomek skoro $\overline{H} / K$, tedy skoro nula, a frekvence, na kterých je jen šum, filtr ztlumí.

@img(wiener.png, 760, Zesílení podle frekvence: rozmazání H klesá k nule, dělení 1/H roste nade všechny meze a Wienerův filtr nejdřív roste a pak klesá)

Gaussovo $H$ je reálné a kladné, sdružené se sebou samým. Rozmazání pohybem má $H$ komplexní a tam na sdružení záleží.

### 🎯 Úkol 4: Wienerův filtr

Napište `wiener(channel, sigma, K)`, kanál bez Gaussova rozmazání se směrodatnou odchylkou `sigma`. Buňka pod funkcí ho použije s $K = 0.01^2$. Nedokončená funkce vrací kanál beze změny.

<details><summary>💡 Nápověda</summary>

1. Spektrum kanálu, $H$ z `gaussian_filter` pro tvar kanálu, vzorec, zpětná transformace a reálná část.
2. Sdružené číslo dává `np.conj`, $|H|^2$ je `np.abs(H) ** 2`.
</details>

<details><summary>🔑 Řešení</summary>

""" + answer(SOLUTION_WIENER) + r"""
</details>
""")

code(WIENER_GIVEN + f"""


fixed = np.dstack([wiener(lin_blurry[..., c], {BLUR_SIGMA}, 0.01 ** 2) for c in range(3)])
doc.new_layer("Deblurred", rgba(linear_to_srgb(fixed))).show()
""", solve=[(WIENER_GIVEN, SOLUTION_WIENER)])

md(r"""
### ✅ Kontrola

Fotka je ostřejší než `astronaut_blur`: jsou vidět oči, pramínky vlasů a švy skafandru. Ostrá jako `astronaut` není, nejjemnější detaily rozmazání ztlumilo pod úroveň šumu a ty se vrátit nedají. Kolem kontrastních hran mohou být slabé vlnky.

> **❓ Otázka**
> Zkuste místo $K = 0.01^2$ hodnoty $0.001^2$ a $0.1^2$. Co se stane a proč?

<details><summary>🔑 Odpověď</summary>

S malým $K$ je filtr blízko dělení $1 / H$: vrací i vysoké frekvence, kde je skoro jen šum, a fotka je zrnitá. S velkým $K$ filtr ztlumí většinu frekvencí, kde $H$ trochu kleslo, a fotka zůstane rozmazaná. $K$ vyvažuje ostrost proti šumu. Správná hodnota závisí na tom, kolik šumu fotka má: ve fotce z mobilu ve tmě víc než v osmibitovém zaokrouhlení.
</details>
""")

# --- 5. NumPy and Krita -------------------------------------------------------
md(r"""
## 5. Do Krity

Kernely v této lekci sloužily ke srovnání, filtr do Krity je celý v NumPy. Má tři režimy, všechny počítají v lineárním RGB:

- `Deblur`: Wienerův filtr s rozmazáním `sigma`.
- `Low pass`: Gaussovo rozmazání `sigma`.
- `High pass`: obrázek minus rozmazaný obrázek plus 0.5, tedy jen detaily kolem šedé. Odečítá se v hodnotách sRGB, tak jak je vrstva ukládá, rozmazaný obrázek už zakódovaný do sRGB.

Posuvník `noise` je odmocnina z $K$, poměr velikosti šumu k velikosti signálu. Posuvník v něm je rovnoměrnější: $K$ od $0.0001$ do $0.01$ je `noise` od 0.01 do 0.1.

### 🎯 Úkol 5: frekvenční filtr

Napište `apply` se třemi režimy. Nedokončená funkce vrací obrázek beze změny.

<details><summary>💡 Nápověda</summary>

1. Převeďte obrázek do lineárního RGB a filtrujte každý ze tří kanálů zvlášť, `Deblur` funkcí `wiener` s $K$ = `noise ** 2`, ostatní režimy funkcí `blur_fft`.
2. Výsledek zakódujte zpět do sRGB.
3. Pro `High pass` odečtěte zakódovaný rozmazaný obrázek od původního `img` a přičtěte 0.5.
4. Alfa kanál zůstává z `img`.
</details>

<details><summary>🔑 Řešení</summary>

""" + answer(SOLUTION_APPLY) + r"""
</details>
""")

code(PARAMS_NP + "\n\n\n" + APPLY_GIVEN + """


sara.live(apply, PARAMS, source="astronaut_blur", target="Filter NumPy")
""", solve=[(APPLY_GIVEN, SOLUTION_APPLY)])

md(r"""
### ✅ Kontrola

Porovnání s filtrem „Goal“ z úvodu. Kontrola počítá s jeho výchozími hodnotami. Pokud jste s nimi hýbali, spusťte buňku „Goal“ znovu. Oba filtry počítají totéž v NumPy, výsledek by se měl lišit nanejvýš zaokrouhlením.
""")

code(f"""
defaults = {DEFAULTS}
sara.check("Goal", apply(blurry.read(), **defaults))
""")

md(r"""
### 🎯 Úkol 6: filtr v Kritě

Vložte filtr do `pga_filter/effect.py` s `TITLE = "Frequency filter"` a vyzkoušejte ho na vlastní fotce. Rozmažte ji v Kritě Gaussovým rozmazáním a zkuste ji režimem `Deblur` zaostřit. Krita měří rozmazání poloměrem, ne směrodatnou odchylkou, takže `sigma` hledejte posuvníkem.

<details><summary>💡 Nápověda</summary>

1. Do `effect.py` patří buňka s pomocnými funkcemi, vaše `gaussian_filter`, `blur_fft`, `wiener`, `PARAMS` a `apply`. Řádek `import numpy as np` nechte nahoře.
2. Filtr dostane výběr, nebo celou vrstvu, a transformace ho bere jako dlaždici. U okraje výběru proto rozmazání přebírá barvy z protějšího okraje.
</details>

> **❓ Otázka**
> Duplikujte vrstvu. Na spodní spusťte `Low pass`, na horní `High pass` se stejnou `sigma` a horní vrstvě nastavte režim prolnutí Grain Merge. Vrstvy dohromady vypadají jako původní obrázek. Proč?

<details><summary>🔑 Odpověď</summary>

Grain Merge sečte vrstvy a odečte 0.5. `High pass` je obrázek minus `Low pass` plus 0.5, takže součet je `Low pass` + obrázek − `Low pass` + 0.5 − 0.5, tedy obrázek, až na zaokrouhlení a ořezání do rozsahu vrstvy. Retušéři tomu říkají frekvenční separace (*frequency separation*): na spodní vrstvě sjednotí barvu pleti a struktura kůže na horní vrstvě zůstane.
</details>
""")

# --- Summary ------------------------------------------------------------------
md(r"""
## Shrnutí

- Obrázek je součet vln. Fourierova transformace dá pro každou frekvenci komplexní číslo, jehož velikost říká, kolik vlny obrázek obsahuje, a úhel, kam je posunutá.
- Spektrum se kreslí s nulovou frekvencí uprostřed (`np.fft.fftshift`) a v logaritmu. Hrany dají čáry kolmo na ně, opakované vzory body, úzké útvary široké spektrum.
- Transformace v kernelu sčítá $N$ hodnot na pixel v každém ze dvou průchodů, práce roste s $N^3$. FFT roste s $N^2 \log N$ a pro velké obrázky vyhraje i s přenosem vrstvy na CPU a zpátky.
- Konvoluce je ve spektru násobení. Rozmazání přes FFT trvá stejně dlouho pro každou $\sigma$, kernel s $\sigma^2$.
- Rozmazání se dá vrátit dělením spektra jen tam, kde je signál silnější než šum. Wienerův filtr to zařídí jedním číslem $K$.
- FFT bere obrázek jako dlaždici, takže okraje se rozmazávají s protějšími a spektrum má přes střed kříž.

### Co jsme vynechali

- **Okna a rozšíření okrajů.** Kříž ve spektru odstraní okno, které obrázek k okrajům plynule ztlumí. Konvoluci dokola zabrání rozšíření obrázku o odraz okrajů před transformací.
- **Algoritmus FFT.** Cooleyho a Tukeyho algoritmus dělí transformaci délky $N$ na dvě poloviční, a tak dál až k délce 1, odtud $N \log N$. Na GPU je to $\log_2 N$ průchodů za sebou.
- **Vzorkování a aliasing.** Zmenšený obrázek má ve spektru kopie původního spektra, a kde se překryjí, vznikne moiré. Proto se obrázek před zmenšením rozmazává.
- **Slepá dekonvoluce.** Tady jste věděli, jakým Gaussem byla fotka rozmazaná. U skutečné fotky rozmazané pohybem nebo špatným zaostřením tvar rozmazání neznáte a musí se odhadnout z fotky samotné.
- **Fourierova transformace v kompresi.** JPEG rozloží každý blok 8 × 8 pixelů kosinovou transformací a vysoké frekvence uloží hrubě, protože je oko vidí nejméně.

### Bonusové úkoly

1. **Fáze a velikost.** Složte obrázek z velikostí spektra fotky a fáze spektra cihel (`imgs/4/brick.png`), $|F_1|\, e^{i \arg F_2}$, a obráceně. `np.angle` dá úhel komplexního čísla. Který z obou obrázků poznáte, a co z toho plyne o tom, kde je ve spektru uložen tvar?
2. **Vrubový filtr** (*notch filter*). Přičtěte k fotce vlnu z kernelu „Wave“ jako pravidelné rušení. Najděte ve spektru její dva body, vynulujte je a fotku transformujte zpátky.
3. **Okraje bez dlaždice.** Rozšiřte kanál před `blur_fft` o odraz okrajů, `np.pad(channel, r, mode="reflect")` s `r` kolem $4\sigma$, a výsledek ořízněte. Porovnejte okraje rozmazané fotky s těmi původními.
4. **Rozmazání pohybem.** Jádro je úsečka délky `length` pixelů pod úhlem `angle`. Nakreslete ji do pole velikosti obrázku se středem v rohu `[0, 0]` (záporné posuny jsou na konci pole), $H$ je `np.fft.fft2` toho pole. $H$ je teď komplexní a `np.conj` ve Wienerově filtru je potřeba.
""")

md(r"""
### Nápady na semestrální práci

- Frekvenční separace jedním příkazem: plugin, který z vrstvy udělá dvojici `Low pass` a `High pass` s režimem Grain Merge, připravenou k retuši.
- Odstranění pravidelného vzoru, třeba rastru z naskenovaného tisku: filtr najde ve spektru jasné body mimo střed a ztlumí je.
- Zaostření rozmazání pohybem s posuvníky délky a úhlu a náhledem jádra.
""")


def build():
    lesson_writer.write(cells, OUT, "l4")

if __name__ == "__main__":
    build()
