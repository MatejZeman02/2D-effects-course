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

GOAL = ("# The filter of section 7, hidden because you write it yourself.\n"
        "def _goal():\n"
        + textwrap.indent("\n\n".join(piece.replace("\n\n\n", "\n\n") for piece in
                                        [CONVERT_NP, SOLUTION_GAUSSIAN, SOLUTION_BLUR, SOLUTION_WIENER, SOLUTION_APPLY]), "    ")
        + "\n\n    return apply\n\n\n"
        + PARAMS_NP
        + '\n\nsara.live(_goal(), PARAMS, source="astronaut_blur", target="Goal")')

DEFAULTS = f'{{"mode": "Deblur", "sigma": {BLUR_SIGMA}, "noise": 0.01}}'


# --- The interactive pictures ---------------------------------------------------
# Hidden cells: what they draw is the lesson, how they draw it is not. Each one
# shows a matplotlib figure under its controls and draws it again on every move.
# No ipywidgets.Output, which VS Code cannot draw: the figure goes to an Image.
TOOLS = '''\
# The tools of the interactive pictures, hidden because they are not part of the lesson.
import io

import ipywidgets as widgets
from IPython.display import display
from matplotlib import font_manager
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from PIL import Image, ImageDraw, ImageFont


class Steps(list):
    """The values of a slider that stops only at them."""


def explore(draw, **controls):
    """draw's figure under a row of controls, drawn again whenever one moves.

    Each control is (label, choices): a list is a drop-down, Steps([...]) a slider over its values,
    a tuple (default, low, high) a slider of whole numbers.
    draw gets the values by the controls' names and answers a Figure, or a Figure and a caption.
    """
    made = {}
    for name, (label, choices) in controls.items():
        if isinstance(choices, Steps):
            made[name] = widgets.SelectionSlider(options=choices, value=choices[0], description=label,
                                                 continuous_update=False)
        elif isinstance(choices, list):
            made[name] = widgets.Dropdown(options=choices, value=choices[0], description=label)
        else:
            default, low, high = choices
            made[name] = widgets.IntSlider(value=default, min=low, max=high, description=label,
                                           continuous_update=False)
        made[name].style.description_width = "initial"
    picture = widgets.Image(format="png", layout=widgets.Layout(max_width="100%"))
    caption = widgets.HTML(layout=widgets.Layout(max_width="60em"))

    def redraw(_=None):
        answer = draw(**{name: control.value for name, control in made.items()})
        fig, text = answer if isinstance(answer, tuple) else (answer, "")
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=72, bbox_inches="tight", facecolor="white")
        picture.value = buffer.getvalue()
        caption.value = text

    for control in made.values():
        control.observe(redraw, names="value")
    redraw()
    display(widgets.VBox([widgets.HBox(list(made.values())), picture, caption]))


def grey_of(path, side):
    """The photo at path, grey, side x side, from 0 to 1."""
    picture = Image.open(path).convert("L").resize((side, side), Image.LANCZOS)
    return np.asarray(picture, dtype=np.float64) / 255


def letter_a(side=256):
    font = ImageFont.truetype(font_manager.findfont("DejaVu Sans:bold"), int(side * 0.55))
    picture = Image.new("L", (side, side), 0)
    ImageDraw.Draw(picture).text((side / 2, side / 2), "A", fill=255, font=font, anchor="mm")
    return np.asarray(picture, dtype=np.float64) / 255


PICTURES = {
    "fotka": grey_of("imgs/4/astronaut.png", 256),
    "písmeno A": letter_a(),
    "cihly": grey_of("imgs/4/brick.png", 256),
}


def log_magnitude(F, thick=False):
    """|F| with zero frequency in the middle, in logarithm, from 0 to 1, with thick each value
    spread to its neighbours, so a lone coefficient shows on screen."""
    m = np.log1p(np.abs(np.fft.fftshift(F)))
    if thick:
        m = np.max([np.roll(m, (dy, dx), axis=(0, 1)) for dy in (-1, 0, 1) for dx in (-1, 0, 1)], axis=0)
    return m / m.max()


def phase_picture(angle, strong):
    """An angle in colour around the circle, grey where the wave is too weak to matter."""
    import matplotlib
    colours = matplotlib.colormaps["twilight"]((np.fft.fftshift(angle) + np.pi) / (2 * np.pi))[..., :3]
    colours[~np.fft.fftshift(strong)] = 0.5
    return colours


def panel(ax, a, title, **keys):
    ax.imshow(a, cmap=keys.pop("cmap", "gray"), **keys)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])'''

PARTS = '''\
# Interactive picture: the four parts of a complex spectrum.
def wave(kind, n=256):
    x = np.arange(n)[None, :] * np.ones((n, 1))
    return 0.5 + 0.5 * (np.cos if kind == "kosinusovka" else np.sin)(2 * np.pi * 6 * x / n)


def signed_log(a):
    """a in logarithm with its sign, from -1 to 1."""
    s = np.sign(a) * np.log1p(np.abs(a))
    return s / max(np.abs(s).max(), 1e-9)


def parts(name, shift):
    f = wave(name) if name in ("kosinusovka", "sinusovka") else PICTURES[name]
    f = np.roll(f, shift, axis=1)
    F = np.fft.fft2(f)
    strong = np.fft.ifftshift(log_magnitude(F)) > 0.4
    views = [(signed_log(np.fft.fftshift(F.real)), "reálná část Re F", {"cmap": "RdBu_r", "vmin": -1, "vmax": 1}),
             (signed_log(np.fft.fftshift(F.imag)), "imaginární část Im F", {"cmap": "RdBu_r", "vmin": -1, "vmax": 1}),
             (log_magnitude(F), "amplituda log |F|", {"vmin": 0, "vmax": 1}),
             (phase_picture(np.angle(F), strong), "fáze arg F, kde je vlna silná", {})]
    zoom = name in ("kosinusovka", "sinusovka")
    middle = slice(128 - 8, 128 + 9) if zoom else slice(None)
    fig = Figure(figsize=(17, 3.8))
    axes = fig.subplots(1, 5)
    panel(axes[0], f, f"{name}, posun o {shift} px", vmin=0, vmax=1)
    for ax, (a, title, keys) in zip(axes[1:], views):
        panel(ax, a[middle, middle], title + (", střed" if zoom else ""), interpolation="nearest", **keys)
    import matplotlib
    turn = matplotlib.cm.ScalarMappable(matplotlib.colors.Normalize(-180, 180), "twilight")
    fig.colorbar(turn, ax=axes[4], ticks=[-180, -90, 0, 90, 180], format="%d°", fraction=0.046, pad=0.04)
    return fig


explore(parts, name=("obrázek", ["kosinusovka", "sinusovka", "fotka", "písmeno A", "cihly"]),
        shift=("posun", (0, 0, 20)))'''


COMPOSE = '''\
# Interactive picture: a picture summed from a few of its waves, low, mid and high.
BANDS = [("nízké", 0, 6, (0.30, 0.47, 0.66)), ("střední", 6, 24, (0.96, 0.52, 0.09)),
         ("vysoké", 24, 1000, (0.33, 0.66, 0.31))]


def compose(picture, count):
    f = PICTURES[picture]
    n = len(f)
    F = np.fft.fft2(f)
    rows, cols = np.indices(F.shape)
    flat = (rows * n + cols).ravel()
    mirror = (((-rows) % n) * n + (-cols) % n).ravel()      # where F(-u, -v) is
    k = np.fft.fftfreq(n, 1 / n)
    distance = np.hypot(k[:, None], k[None, :]).ravel()     # periods per picture
    one_of_pair = flat[(flat <= mirror) & (flat > 0)]        # F(u, v) and F(-u, -v) are one real wave
    fig = Figure(figsize=(18, 3.6))
    axes = fig.subplots(1, 6)
    shown = np.dstack([0.3 * log_magnitude(F)] * 3)
    total = np.full((n, n), F[0, 0].real / n ** 2)          # the mean, the wave of frequency 0
    used = 1
    for i, (name, low, high, colour) in enumerate(BANDS):
        band = one_of_pair[(distance[one_of_pair] > low) & (distance[one_of_pair] <= high)]
        strongest = band[np.argsort(-np.abs(F.ravel()[band]), kind="stable")][:count]
        mask = np.zeros(n * n, bool)
        mask[strongest] = mask[mirror[strongest]] = True
        mask = mask.reshape(n, n)
        part = np.fft.ifft2(F * mask).real
        total += part
        used += len(strongest)
        shown[np.fft.fftshift(mask)] = colour
        ys, xs = np.nonzero(np.fft.fftshift(mask))
        axes[0].scatter(xs, ys, s=7, color=colour, linewidths=0)
        top = np.abs(part).max() or 1
        panel(axes[i + 1], part, f"{name}: {len(strongest)} z {len(band)} vln", vmin=-top, vmax=top)
        for side in "top", "bottom", "left", "right":
            axes[i + 1].spines[side].set(color=colour, linewidth=3)
    panel(axes[0], shown, "střed spektra, vybrané vlny barevně")
    axes[0].set_xlim(n // 2 - 64, n // 2 + 64)            # the middle quarter, where the chosen waves lie
    axes[0].set_ylim(n // 2 + 64, n // 2 - 64)
    for _, _, high, colour in BANDS[:2]:                   # the bands' borders
        axes[0].add_patch(Circle((n // 2, n // 2), high, fill=False, color=colour, lw=1))
    panel(axes[4], total, f"součet: {used} vln, {100 * used / len(one_of_pair):.2g} % ze všech", vmin=0, vmax=1)
    panel(axes[5], f, "obrázek", vmin=0, vmax=1)
    return fig


explore(compose, count=("vln v pásmu", Steps([1, 2, 3, 5, 10, 20, 50, 100, 200, 500])),
        picture=("obrázek", list(PICTURES)))'''

SPECTRA = '''\
# Interactive picture: pictures and their spectra.
CAPTIONS = {
    "konstanta": "Konstanta nemá žádnou vlnu kromě nulové frekvence. Spektrum je jediný bod uprostřed, "
                 "<b>DC složka</b>, součet všech pixelů.",
    "bod": "Jediný bod obsahuje všechny frekvence stejně silně, spektrum je konstanta. Je to opak konstanty: "
           "co je v obrázku úzké, je ve spektru široké.",
    "kosinusovka": "Jedna vlna dá ve spektru <b>dva</b> body souměrně kolem středu, a k tomu DC složku, protože "
                   "vlna jde od 0 do 1 kolem šedé. Posuvník zvyšuje frekvenci a body se vzdalují od středu.",
    "šikmá kosinusovka": "Body leží ve směru, ve kterém se vlna mění, tedy <b>kolmo na pruhy</b>.",
    "obdélník": "Obdélník dá funkci sinc, sin(x)/x: kříž s vlnkami. Užší strana obdélníku dá širší rameno kříže.",
    "kruh": "Kruh dá soustředné kroužky (Besselova funkce), kruhovou obdobu funkce sinc.",
    "Gauss": "Gauss dá zase Gauss, bez vlnek. Čím širší v obrázku, tím užší ve spektru. Proto je Gauss "
             "dobrý filtr.",
    "hřeben bodů": "Body s roztečí T dají ve spektru body s roztečí N / T. Hustší mřížka v obrázku, "
                   "řidší ve spektru.",
    "Gabor": "Gauss krát kosinus dá dva Gaussy posunuté na frekvenci kosinu. Násobení vlnou spektrum posune.",
    "hrana": "Svislá hrana se mění podél x, a tak dá ve spektru <b>vodorovnou</b> čáru, kolmo na sebe.",
    "fotka": "Spektrum fotky je nejsilnější uprostřed: velké plochy nesou víc energie než detaily. "
             "Kříž přes střed dělají okraje fotky, zkuste okno.",
    "písmeno A": "Šikmé hrany písmene dají ve spektru šikmé čáry, každou kolmo na svou hranu.",
    "cihly": "Opakovaný vzor dá mřížku jasných bodů, jako hřeben.",
}


def pattern(name, size, n=256):
    y, x = np.indices((n, n)) - n // 2
    r = np.hypot(x, y)
    if name in PICTURES:
        return PICTURES[name]
    if name == "konstanta":
        return np.full((n, n), 0.5)
    if name == "bod":
        return (r == 0).astype(float)
    if name == "kosinusovka":
        return 0.5 + 0.5 * np.cos(2 * np.pi * 2 * size * x / n)
    if name == "šikmá kosinusovka":
        return 0.5 + 0.5 * np.cos(2 * np.pi * (2 * size * x + size * y) / n)
    if name == "obdélník":
        return ((np.abs(x) < size) & (np.abs(y) < 3 * size)).astype(float)
    if name == "kruh":
        return (r < 3 * size).astype(float)
    if name == "Gauss":
        return np.exp(-r ** 2 / (2 * (2 * size) ** 2))
    if name == "hřeben bodů":
        T = [4, 8, 16, 32][min(3, (size - 1) // 3)]
        return ((x % T == 0) & (y % T == 0)).astype(float)
    if name == "Gabor":
        return np.exp(-r ** 2 / (2 * (2 * size) ** 2)) * np.cos(2 * np.pi * 24 * x / n)
    return (x >= 0).astype(float)                         # hrana


def spectra(name, size, window):
    f = pattern(name, size)
    n = len(f)
    if window == "Hannovo":
        f = f * np.outer(np.hanning(n), np.hanning(n))
    F = np.fft.fft2(f)
    fig = Figure(figsize=(14, 4.2))
    axes = fig.subplots(1, 3, gridspec_kw={"width_ratios": [1, 1, 1.5]})
    panel(axes[0], f, "obrázek")
    panel(axes[1], log_magnitude(F, thick=True), "spektrum, log |F|", vmin=0, vmax=1)
    u = np.arange(n) - n // 2
    axes[2].semilogy(u, np.abs(np.fft.fftshift(F))[n // 2] + 1e-3, color="#F58518")
    axes[2].set_title("řez spektrem vodorovně přes střed, |F(u, 0)|", fontsize=10)
    axes[2].set_xlabel("u")
    return fig, CAPTIONS[name]


explore(spectra, name=("obrázek", list(CAPTIONS)), size=("velikost", (3, 1, 10)),
        window=("okno", ["žádné", "Hannovo"]))'''

PROPERTIES = '''\
# Interactive picture: what a change of the picture does to the spectrum.
CHANGES = ["posun", "otočení", "změna měřítka", "rozmazání (konvoluce)", "násobení vlnou", "součet s vlnou",
           "jen amplituda", "jen fáze", "amplituda z A, fáze z fotky", "amplituda z fotky, fáze z A"]



def changed(f, change, amount):
    """f after the change, and what the change was, in words."""
    n = len(f)
    y, x = np.indices(f.shape)
    F = np.fft.fft2(f)
    if change == "posun":
        return np.roll(f, (2 * amount, 3 * amount), axis=(0, 1)), f"posun o ({3 * amount}, {2 * amount}) px"
    if change == "otočení":
        turned = Image.fromarray(f.astype(np.float32)).rotate(9 * amount, Image.BILINEAR)
        return np.asarray(turned, dtype=np.float64), f"otočení o {9 * amount}°"
    if change == "změna měřítka":
        a = 2 ** ((amount - 5) / 5)
        c = n / 2
        grown = Image.fromarray(f.astype(np.float32)).transform(
            (n, n), Image.AFFINE, (1 / a, 0, c - c / a, 0, 1 / a, c - c / a), Image.BILINEAR)
        return np.asarray(grown, dtype=np.float64), f"měřítko {a:.2f}"
    if change == "rozmazání (konvoluce)":
        sigma = 0.6 * amount
        fy, fx = np.fft.fftfreq(n)[:, None], np.fft.fftfreq(n)[None, :]
        H = np.exp(-2 * np.pi ** 2 * sigma ** 2 * (fx ** 2 + fy ** 2))
        return np.fft.ifft2(F * H).real, f"Gauss se σ = {sigma:.1f} px"
    if change == "násobení vlnou":
        return f * np.cos(2 * np.pi * 8 * amount * x / n), f"krát kosinus s u = {8 * amount}"
    if change == "součet s vlnou":
        return f + 0.3 * np.cos(2 * np.pi * 8 * amount * x / n), f"plus kosinus s u = {8 * amount}"
    if change == "jen amplituda":
        return np.fft.fftshift(np.fft.ifft2(np.abs(F)).real), "amplituda, fáze všude 0 (posunuto doprostřed)"
    if change == "jen fáze":
        return np.fft.ifft2(np.exp(1j * np.angle(F))).real, "fáze, amplituda všude 1"
    P = np.fft.fft2(PICTURES["fotka"])
    if change == "amplituda z A, fáze z fotky":
        return np.fft.ifft2(np.abs(F) * np.exp(1j * np.angle(P))).real, "amplituda A, fáze fotky"
    return np.fft.ifft2(np.abs(P) * np.exp(1j * np.angle(F))).real, "amplituda fotky, fáze A"


def properties(change, amount):
    f = PICTURES["písmeno A"]
    g, what = changed(f, change, amount)
    fig = Figure(figsize=(14, 7.4))
    axes = fig.subplots(2, 4)
    spectra = []
    for row, (picture, title) in enumerate(((f, "písmeno A"), (g, what))):
        F = np.fft.fft2(picture)
        spectra.append(F)
        strong = np.fft.ifftshift(log_magnitude(F)) > 0.4
        lo, hi = np.percentile(picture, [0.5, 99.5])
        panel(axes[row, 0], picture, title, vmin=lo, vmax=hi if hi > lo else lo + 1)
        panel(axes[row, 1], log_magnitude(F, thick=True), "amplituda, log |F|", vmin=0, vmax=1)
        panel(axes[row, 2], phase_picture(np.angle(F), strong), "fáze arg F, kde je vlna silná")
    F, G = spectra
    both = (np.fft.ifftshift(log_magnitude(F)) > 0.4) & (np.fft.ifftshift(log_magnitude(G)) > 0.4)
    panel(axes[1, 2 + 1], phase_picture(np.angle(G * np.conj(F)), both), "změna fáze proti A")
    axes[0, 3].axis("off")
    energy = (g ** 2).sum()
    fig.suptitle(f"Parseval: Σ|f|² = {energy:.1f}, Σ|F|² / N² = {(np.abs(np.fft.fft2(g)) ** 2).sum() / g.size:.1f}",
                 fontsize=10, y=0.02)
    return fig


explore(properties, change=("změna", CHANGES), amount=("míra", (4, 0, 10)))'''

SAMPLING = '''\
# Interactive picture: sampling and aliasing.
def zone_plate(n=512):
    y, x = np.indices((n, n)) - n // 2
    return 0.5 + 0.5 * np.cos(np.pi * (x ** 2 + y ** 2) / n)     # frequency grows with the distance


def sampling(frequency, factor):
    fig = Figure(figsize=(13, 7.6))
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 1.5], hspace=0.35)
    ax = fig.add_subplot(grid[0, :])
    t = np.linspace(0, 1, 2000)
    samples = np.arange(16) / 16
    alias = abs(frequency - 16 * round(frequency / 16))
    ax.plot(t, np.cos(2 * np.pi * frequency * t), color="#9a9a9a", lw=1.2, label=f"vlna, {frequency} period")
    ax.plot(samples, np.cos(2 * np.pi * frequency * samples), "o", color="#F58518", label="16 vzorků")
    if alias != frequency:
        ax.plot(t, np.cos(2 * np.pi * alias * t), "--", color="#4C78A8", label=f"vzorky vypadají jako {alias} period")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.08), ncol=3, fontsize=9, frameon=False)
    ax.set_yticks([])

    f = zone_plate()
    n = len(f)
    fy, fx = np.fft.fftfreq(n)[:, None], np.fft.fftfreq(n)[None, :]
    sigma = factor / 2                                   # cuts what the smaller picture cannot hold
    soft = np.fft.ifft2(np.fft.fft2(f) * np.exp(-2 * np.pi ** 2 * sigma ** 2 * (fx ** 2 + fy ** 2))).real
    side = n // factor
    panel(fig.add_subplot(grid[1, 0]), f, "zónová deska, 512 × 512", vmin=0, vmax=1)
    panel(fig.add_subplot(grid[1, 1]), f[::factor, ::factor], f"každý {factor}. pixel, {side} × {side}",
          vmin=0, vmax=1, interpolation="nearest")
    panel(fig.add_subplot(grid[1, 2]), soft[::factor, ::factor], f"rozmazat, pak každý {factor}. pixel",
          vmin=0, vmax=1, interpolation="nearest")
    return fig


explore(sampling, frequency=("frekvence", (5, 1, 15)), factor=("zmenšení", [4, 2, 8]))'''

FILTERS = '''\
# Interactive picture: filters as masks on the spectrum.
FILTERS = ["dolní propust, ideální", "dolní propust, Gauss", "dolní propust, Butterworth",
           "horní propust, Gauss", "pásmová propust, Gauss"]
photo_grey = grey_of("imgs/4/astronaut.png", 512)


def mask(kind, radius, n):
    fy, fx = np.fft.fftfreq(n, 1 / n)[:, None], np.fft.fftfreq(n, 1 / n)[None, :]
    d = np.hypot(fx, fy)                                 # distance from zero frequency, in periods per picture
    if kind == "dolní propust, ideální":
        return (d <= radius).astype(float)
    if kind == "dolní propust, Gauss":
        return np.exp(-d ** 2 / (2 * radius ** 2))
    if kind == "dolní propust, Butterworth":
        return 1 / (1 + (d / radius) ** 4)
    if kind == "horní propust, Gauss":
        return 1 - np.exp(-d ** 2 / (2 * radius ** 2))
    return np.exp(-(d - radius) ** 2 / (2 * (radius / 3) ** 2))


def filters(kind, radius):
    f = photo_grey
    H = mask(kind, radius, len(f))                       # in np.fft.fft2's order, zero frequency in the corner
    F = np.fft.fft2(f)
    g = np.fft.ifft2(F * H).real
    if not kind.startswith("dolní"):
        g = g + 0.5                                      # detail around grey
    fig = Figure(figsize=(15, 4.1))
    axes = fig.subplots(1, 4)
    panel(axes[0], np.fft.fftshift(H), "maska H", vmin=0, vmax=1)
    panel(axes[1], log_magnitude(F * H) * (np.fft.fftshift(H) > 1e-3), "spektrum krát H", vmin=0, vmax=1)
    panel(axes[2], g, "výsledek", vmin=0, vmax=1)
    panel(axes[3], g[150:300, 180:330], "výřez", vmin=0, vmax=1, interpolation="nearest")
    return fig


explore(filters, kind=("filtr", FILTERS), radius=("poloměr", (30, 2, 120)))'''


def answer(snippet, lang="python"):
    """A solution block's code fence."""
    return f"```{lang}\n{snippet}\n```"


# --- Title --------------------------------------------------------------------
md(r"""
# Lekce 4: Frekvenční oblast

Na konci lekce budete mít filtr, který rozmazanou fotku zaostří. Rozmazání se nedá vrátit pixel po pixelu, ve spektru obrázku je to ale obyčejné násobení, a to se vrátit dá. Nejdřív si na interaktivních obrázcích ukážete, co spektrum je a jak ho číst. Pak spočítáte Fourierovu transformaci na GPU i v NumPy a fotku přes frekvence rozmažete a zaostříte.

Spusťte obě buňky. První otevře v Sáře nový dokument s plátnem 512 × 512 a fotky jako vrstvy. Fotka `astronaut_blur.png` je ostrá fotka `astronaut.png` rozmazaná Gaussem. Druhá buňka připraví interaktivní obrázky. Ty mají kód schovaný: stačí buňku spustit a hýbat nabídkami a posuvníky nad obrázkem. Kdo chce vědět, jak se kreslí, buňku si rozbalí.
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

code(TOOLS, hidden=True)

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
| `const int N = 512` | celočíselná konstanta | 4 |
| `a % b` | zbytek po dělení celých čísel, u `ivec2` po složkách | 4, 5 |
| `vec2` jako komplexní číslo | reálná část v `x`, imaginární v `y` | 4 |
| `exp(x)`, `ceil(x)` | $e^x$ a zaokrouhlení nahoru | 5 |

**NumPy**

| Volání | Co dělá | Sekce |
|---|---|---|
| `np.fft.fft2(a)`, `np.fft.ifft2(F)` | dvourozměrná Fourierova transformace a zpětná transformace | 1, 5 |
| `np.fft.fftshift(F)` | přesune nulovou frekvenci doprostřed pole | 1 |
| `np.abs(F)`, `F.real`, `np.conj(F)` | absolutní hodnota, reálná část a komplexně sdružené číslo, po prvcích | 1, 5, 6 |
| `np.log1p(x)` | $\ln(1 + x)$ po prvcích | 1 |
| `1j` | imaginární jednotka | 4 |
| `np.fft.fftfreq(n)` | frekvence `n` koeficientů v pořadí, v jakém je vrací `np.fft.fft2` | 5 |
""")

md(r"""
## Cíl lekce

Buňka níže spustí hotový filtr na rozmazanou fotku. Jeho kód je schovaný, protože ho během lekce napíšete sami. Režim `Deblur` fotku zaostří, `Low pass` rozmaže a `High pass` nechá jen detaily. Vyzkoušejte, co dělá posuvník `noise` u režimu `Deblur`, hlavně blízko nuly.
""")

code(GOAL, hidden=True)

# --- 1. A picture as a sum of waves ---------------------------------------------
md(r"""
## 1. Obrázek jako součet vln

Každý obrázek se dá zapsat jako součet **vln**, obrázků s průběhem kosinu. Vlna má **frekvenci** $(u, v)$: na šířku obrázku se do ní vejde $u$ period a na výšku $v$ period. Kromě frekvence má **amplitudu**, jak je silná, a **fázi**, o kolik je posunutá. **Fourierova transformace** k obrázku najde právě ty vlny, ze kterých se skládá. Je to tatáž informace, jen jinak zapsaná: převod nic neztratí a zpětná transformace vrátí původní obrázek přesně.

Obrázek níže skládá fotku z několika jejích vln. Vlny jsou rozdělené podle frekvence do tří pásem: **nízké** (modře, do 6 period na obrázek), **střední** (oranžově, do 24 period) a **vysoké** (zeleně). Z každého pásma se berou jen ty nejsilnější. Posuvník `vln v pásmu` začíná na jedné vlně z každého pásma. Posouvejte ho doprava a sledujte, co které pásmo do součtu přidá a kde jeho vlny leží ve spektru vlevo.
""")

code(COMPOSE, hidden=True)

md(r"""
Nízké vlny nesou velké plochy a celkový jas, střední tvary, vysoké hrany a detaily. S pěti sty vlnami v každém pásmu, asi třemi procenty všech vln, je fotka čitelná. Tak pracuje ztrátová komprese, třeba JPEG: uloží silné vlny a slabé zahodí.

### Jak transformace vlny najde

Pro každou frekvenci $(u, v)$ obrázek vynásobí vlnou té frekvence, pixel po pixelu, a všechno sečte. Je to skalární součin obrázku s vlnou: čím víc se obrázek vlně podobá, tím větší součet. Pro obrázek $f$ o straně $N$ pixelů:

$$F(u, v) = \sum_{x=0}^{N-1} \sum_{y=0}^{N-1} f(x, y)\, e^{-2\pi i\,(ux + vy)/N}$$

Podle Eulerova vzorce je $e^{-i\varphi} = \cos\varphi - i \sin\varphi$, takže suma jsou dva součty najednou, s kosinem a se sinem téže frekvence. Kosinus sám by vlnu posunutou o čtvrt periody nenašel, jejich součin má součet nula, a tu najde sinus. Oba součty dají komplexní číslo $F(u, v)$. Jeho absolutní hodnota $|F|$ je amplituda vlny, jeho úhel $\arg F$ je její fáze.

@img(dot_product.png, 900, Malá fotka vynásobená kosinem a minus sinem frekvence 2 a 1, oba součiny pixel po pixelu, jejich součty jako reálná a imaginární část F a šipka komplexního čísla F s amplitudou a fází)

Obrázek $f(x, y)$, hodnoty pixelů, je **prostorová oblast**. Komplexní čísla $F(u, v)$ jsou **frekvenční oblast**. Obrázek amplitud $|F|$ pro všechny frekvence je **spektrum**.

Kernel níže nakreslí do Sáry jednu vlnu. Posuvníky `u` a `v` mění frekvenci, `phase` fázi.
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
Spektrum je komplexní a kreslit se dá po částech. Obrázek níže ukazuje všechny čtyři vedle sebe:

- **Reálná část** $\mathrm{Re}\,F$ je skalární součin obrázku s kosinem, **imaginární** $\mathrm{Im}\,F$ se sinem se znaménkem minus. Červená je kladná, modrá záporná, v logaritmu se znaménkem.
- **Amplituda** $|F| = \sqrt{\mathrm{Re}^2 + \mathrm{Im}^2}$ je délka šipky z obrázku se skalárním součinem, **fáze** $\arg F$ její úhel.

Kosinusovka má spektrum jen reálné, sinusovka jen imaginární. Reálná část je souměrná podle středu, imaginární má na protějších bodech opačné znaménko. Posuvník `posun` posune obrázek doprava: reálná a imaginární část se přelévají jedna do druhé a fáze se mění, ale amplituda zůstane stejná. U fotky jsou reálná i imaginární část změť znamének a čitelná je jen amplituda. Proto se jako spektrum obvykle kreslí amplituda.
""")

code(PARTS, hidden=True)

# --- 2. Reading a spectrum ---------------------------------------------------------
md(r"""
## 2. Jak číst spektrum

- **Střed** je nulová frekvence, **DC složka** $F(0, 0)$: součet všech pixelů, tedy průměrný jas krát počet pixelů.
- **Blízko středu** jsou nízké frekvence, velké plochy a celkový tvar. **Daleko od středu** jsou vysoké, hrany, detaily a šum.
- **Směr:** pruhy a hrany dají ve spektru body nebo čáru **kolmo** na sebe. Svislá hrana se mění podél $x$, a tak dá vodorovnou čáru.
- **Souměrnost:** obrázek má reálná čísla, a proto je $F(-u, -v)$ komplexně sdružené s $F(u, v)$. Spektrum je souměrné podle středu a jedna kosinusovka dá dva body.
- **Logaritmus:** DC složka je o mnoho řádů větší než ostatní, spektrum se proto kreslí jako $\ln(1 + |F|)$.

Obrázek níže ukazuje dvojice obrázek a spektrum. Vyberte obrázek v nabídce a posuvníkem `velikost` ho zužujte a rozšiřujte. Graf vpravo je vodorovný řez spektrem přes střed, v logaritmu. Pod obrázkem je vysvětlení dvojice.
""")

code(SPECTRA, hidden=True)

md(r"""
> **❓ Otázka**
> Vyberte fotku. Spektrum má přes střed světlou vodorovnou a svislou čáru, i když fotka žádné takové hrany nemá. Odkud se vzaly, a proč s Hannovým oknem zmizí?

<details><summary>🔑 Odpověď</summary>

Transformace počítá s obrázkem, jako by se opakoval dokola jako dlaždice: za pravým okrajem pokračuje levý, pod dolním horní. Okraje fotky na sebe nenavazují, a kde se potkají, vznikne skok, svislá hrana po celé výšce a vodorovná po celé šířce. Ty dají ve spektru čáry kolmo na ně. Hannovo okno fotku k okrajům plynule ztlumí k nule, dlaždice pak na sebe navazují a skok zmizí.
</details>

### Vlastnosti

| Vlastnost | Obrázek | Spektrum |
|---|---|---|
| linearita | $a f + b g$ | $a F + b G$ |
| posun | $f(x - x_0, y - y_0)$ | $F \cdot e^{-2\pi i (u x_0 + v y_0)/N}$: amplituda stejná, mění se jen fáze |
| otočení | o úhel $\alpha$ | spektrum se otočí o tentýž úhel |
| změna měřítka | $f(ax, ay)$ | přibližně $\frac{1}{a^2} F(u / a, v / a)$: co se v obrázku zúží, ve spektru se roztáhne |
| konvoluce | $f * g$ | $F \cdot G$ |
| násobení | $f \cdot g$ | $\frac{1}{N^2}\, F * G$, konvoluce spekter |
| oddělitelnost | $f(x, y) = g(x)\, h(y)$ | $F(u, v) = G(u)\, H(v)$. 2D transformace je 1D transformace po řádcích a pak po sloupcích |
| Parsevalova rovnost | $\sum \lvert f \rvert^2$ | $\frac{1}{N^2} \sum \lvert F \rvert^2$, energie se zachová |

Obrázek níže změní písmeno A a ukáže, co se stane se spektrem. Horní řádek je původní A, dolní změněné. Ve sloupci fáze je úhel vyjádřený barvou, šedé jsou slabé vlny, na jejichž fázi nezáleží. Poslední sloupec je rozdíl fáze změněného a původního A, tmavá barva znamená beze změny. Pod obrázkem je Parsevalova rovnost pro změněný obrázek.
""")

code(PROPERTIES, hidden=True)

md(r"""
Co si z obrázku odnést:

- **Posun** amplitudu nezmění vůbec, jen fázi, a to úhlem, který s frekvencí rovnoměrně roste: rozdíl fází jsou rovné pruhy.
- **Konvoluce je ve spektru násobení.** Rozmazání vynásobí spektrum Gaussem, který od středu klesá, a vysoké frekvence ztlumí. Je to nejdůležitější vlastnost lekce: filtr, který je v obrázku drahá konvoluce, je ve spektru obyčejné násobení. Stojí na ní sekce 5 a 6.
- **Fáze nese tvar.** Obrázek s amplitudou fotky a fází písmene A vypadá jako A, a obráceně. Fáze říká, kde jsou hrany, amplituda hlavně kolik čeho v obrázku je.
""")

# --- 3. Sampling and aliasing ------------------------------------------------------
md(r"""
## 3. Vzorkování a aliasing

Obrázek je vzorkovaný: z plynulého světla zná jen hodnotu v každém pixelu. Ve spektru to znamená, že se spektrum periodicky opakuje, a když jsou kopie příliš blízko, překryjí se. Vysoká frekvence se pak tváří jako nízká. Tomu se říká **aliasing**: moiré na proužkované košili v televizi, schody na šikmých hranách nebo kola auta ve filmu, která se točí pozpátku.

**Nyquistova věta:** vzorků musí být víc než dvakrát tolik, kolik má signál period. Obrázek o $N$ pixelech unese nejvýš $N / 2$ period, střídání černá, bílá, černá. Obrázek níže nahoře vzorkuje vlnu šestnácti vzorky. Posuvníkem `frekvence` přejděte přes 8 a sledujte, jakou vlnu vzorky ukazují. Dole je zónová deska, kruhy s frekvencí rostoucí od středu, zmenšená tak, že se bere každý $k$-tý pixel.
""")

code(SAMPLING, hidden=True)

md(r"""
Zmenšení výběrem pixelů vyrobí kruhy, které v původním obrázku nejsou: frekvence nad novou Nyquistovou mezí se přeloží na nízké. Obrana je jediná, odstranit před zmenšením frekvence, které menší obrázek neunese, tedy obrázek rozmazat. Proto dobré programy obrázek před zmenšením filtrují.
""")

# --- 4. The DFT in a kernel ---------------------------------------------------
md(r"""
## 4. Fourierova transformace v kernelu

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

Vzorec ze sekce 1 je **diskrétní Fourierova transformace** (DFT): $N$ vzorků dá $N$ frekvencí. Spočítaná přímo potřebuje pro každou z $N$ frekvencí $N$ násobení, v jednom rozměru $N^2$ operací. **FFT** je algoritmus, který dá přesně tentýž výsledek za $N \log_2 N$ operací: rozdělí vzorky na sudé a liché, spočítá dvě transformace poloviční délky a složí je, a tak dál až k délce 1.

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

# --- 5. Filtering in frequency ------------------------------------------------
md(r"""
## 5. Filtrování ve frekvencích

Ze sekce 2 víte, že konvoluce obrázku s jádrem je ve spektru násobení (**věta o konvoluci**). Spektrum rozmazaného obrázku je spektrum obrázku krát spektrum jádra, koeficient po koeficientu:

$$\mathcal{F}\{f * g\} = \mathcal{F}\{f\} \cdot \mathcal{F}\{g\}$$

Spektrum Gaussova jádra se směrodatnou odchylkou $\sigma$ pixelů je opět Gauss, tím užší, čím je jádro širší:

$$H(f_x, f_y) = e^{-2\pi^2 \sigma^2 (f_x^2 + f_y^2)}$$

$f_x$ a $f_y$ jsou frekvence v periodách na pixel, $u / N$ a $v / N$, od $-0.5$ do $0.5$. $H$ je u nulové frekvence 1 a k vysokým frekvencím klesá k nule. Rozmazání tedy ztlumí jemné detaily a velké plochy nechá, je to **dolní propust** (*low pass*).

@img(convolution.png, 900, Nahoře konvoluce fotky s Gaussovým jádrem, dole táž operace ve spektru jako násobení spektra fotky spektrem jádra)

Rozmazání je jeden z filtrů, které spektrum násobí maskou $H$. Postup je vždy stejný: transformace obrázku, násobení maskou, zpětná transformace a reálná část. Obrázek níže zkouší několik masek:

- **Dolní propust** (*low pass*) nechá nízké frekvence a výsledek je rozmazaný.
- **Horní propust** (*high pass*), třeba $1 - H$ Gaussovy dolní propusti, nechá jen hrany a detaily.
- **Pásmová propust** (*band pass*) nechá jen prstenec frekvencí.
- **Ideální** dolní propust, uvnitř kruhu 1 a venku 0, udělá kolem hran vlnky (*ringing*). Ostrý okraj masky je v obrázku funkce sinc, jako obdélník v sekci 2. Proto se používají hladké masky, Gaussova nebo Butterworthova.

Maska musí mít nulovou frekvenci tam, kde ji má spektrum. Obrázek masku počítá rovnou s frekvencemi z `np.fft.fftfreq`, ve stejném pořadí jako spektrum, a jen ji kreslí se středem uprostřed. Masku nakreslenou se středem uprostřed by před násobením vrátil do pořadí spektra `np.fft.ifftshift`.
""")

code(FILTERS, hidden=True)

md(r"""
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

# --- 6. Wiener deconvolution --------------------------------------------------
md(r"""
## 6. Wienerova dekonvoluce

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

# --- 7. NumPy and Krita -------------------------------------------------------
md(r"""
## 7. Do Krity

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

- Obrázek je součet vln. Fourierova transformace dá pro každou frekvenci komplexní číslo: amplituda $|F|$ říká, kolik té vlny obrázek obsahuje, fáze $\arg F$, kam je posunutá. Převod nic neztratí.
- $F(u, v)$ je skalární součin obrázku s vlnou, součet s kosinem a se sinem téže frekvence.
- Spektrum se kreslí s nulovou frekvencí uprostřed (`np.fft.fftshift`) a v logaritmu.
- Transformace v kernelu sčítá $N$ hodnot na pixel v každém ze dvou průchodů, práce roste s $N^3$. FFT dá tentýž výsledek za $N^2 \log N$ a pro velké obrázky vyhraje i s přenosem vrstvy na CPU a zpátky.
- Filtr ve frekvencích je násobení spektra maskou. Rozmazání přes FFT trvá stejně dlouho pro každou $\sigma$, kernel s $\sigma^2$.
- Rozmazání se dá vrátit dělením spektra jen tam, kde je signál silnější než šum. Wienerův filtr to zařídí jedním číslem $K$.

### Na co si dát pozor

- Posun mění jen fázi, amplitudu ne.
- Co je v obrázku úzké, je ve spektru široké, a naopak.
- Konvoluce v obrázku je násobení ve spektru.
- Kosinusovka dá dva body, ne jeden.
- Aliasing nastane, když vzorků není víc než dvakrát tolik, kolik má signál period.
- Ostrá, ideální maska dělá kolem hran vlnky.
- Transformace bere obrázek jako dlaždici: okraje se rozmazávají s protějšími a spektrum má přes střed kříž.

### Dvojice, které je dobré znát

| Obrázek | Spektrum |
|---|---|
| konstanta | jeden bod, DC složka |
| bod | konstanta, všechny frekvence |
| kosinusovka | dva body souměrně kolem středu |
| obdélník | sinc, kříž s vlnkami |
| kruh | soustředné kroužky (Besselova funkce) |
| Gauss | Gauss, tím užší, čím je obrázek širší |
| hřeben bodů s roztečí $T$ | hřeben bodů s roztečí $N / T$ |
| Gauss krát kosinus (Gabor) | dva Gaussy na frekvenci kosinu |

### Co jsme vynechali

- **Rozšíření okrajů.** Konvoluci dokola zabrání rozšíření obrázku o odraz okrajů před transformací.
- **FFT na GPU.** Cooleyho a Tukeyho algoritmus je na GPU $\log_2 N$ průchodů za sebou, jak to dělají knihovny VkFFT a cuFFT.
- **Slepá dekonvoluce.** Tady jste věděli, jakým Gaussem byla fotka rozmazaná. U skutečné fotky rozmazané pohybem nebo špatným zaostřením tvar rozmazání neznáte a musí se odhadnout z fotky samotné.
- **Fourierova transformace v kompresi.** JPEG rozloží každý blok 8 × 8 pixelů kosinovou transformací a vysoké frekvence uloží hrubě, protože je oko vidí nejméně.

### Bonusové úkoly

1. **Vrubový filtr** (*notch filter*). Přičtěte k fotce vlnu z kernelu „Wave“ jako pravidelné rušení. Najděte ve spektru její dva body, vynulujte je a fotku transformujte zpátky.
2. **Okraje bez dlaždice.** Rozšiřte kanál před `blur_fft` o odraz okrajů, `np.pad(channel, r, mode="reflect")` s `r` kolem $4\sigma$, a výsledek ořízněte. Porovnejte okraje rozmazané fotky s těmi původními.
3. **Rozmazání pohybem.** Jádro je úsečka délky `length` pixelů pod úhlem `angle`. Nakreslete ji do pole velikosti obrázku se středem v rohu `[0, 0]` (záporné posuny jsou na konci pole), $H$ je `np.fft.fft2` toho pole. $H$ je teď komplexní a `np.conj` ve Wienerově filtru je potřeba.
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
