"""Builds the lesson 0 notebook in the lecture folder above this one.

The notebook is generated: edit this file and run it again, never the .ipynb
by hand, or the next build overwrites the edit. The pictures come from
lesson_00_diagrams.py.

Markdown cells are raw strings, so LaTeX keeps single backslashes and braces.
A picture from imgs/0 is written as @img(file, width, alt text).
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "lesson_00_canvas_kernel_krita.ipynb"
IMG = "imgs/0"
cells = []


def _pictures(text):
    return re.sub(r"@img\(([^,]+), (\d+), ([^)]*)\)",
                  lambda m: f'<img src="{IMG}/{m[1]}" width="{m[2]}" alt="{m[3]}">', text)


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": _pictures(text).strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(keepends=True)})


# --- Title --------------------------------------------------------------------
md(r"""
# Lekce 0: Plátno, kernel a Krita

V této lekci si připravíte nástroje pro celý blok 2D grafiky a vyzkoušíte je na inverzi barev a převodu do šedi. Na konci budete mít:

- **Sáru**, malířský program počítající na grafické kartě, propojenou s tímto notebookem,
- první **kernel**, krátký program pro GPU, který přepočítá každý pixel obrázku,
- stejné efekty v **NumPy**,
- **plugin do Krity**, ve kterém poběží tatáž NumPy funkce.

Stejný postup má každá lekce:

@img(course_loop.png, 860, Kernel v gmacs, funkce v NumPy, plugin v Kritě)

Semestrální práce je plugin do Krity. Efekt se ladí v notebooku, kde se kernel po pohybu posuvníkem hned přepočítá. Hotový efekt se přepíše do NumPy a vloží do pluginu.
""")

md(r"""
### Jak číst tento notebook

- Buňky spouštějte popořadě (`Shift + Enter`). Většina je hotová, stačí ji spustit.
- **🎯 Úkol** je místo pro vaši práci. Pod ním je rozbalovací **💡 Nápověda** a zvlášť **🔑 Řešení**, které otevřete až po vlastním pokusu.
- **✅ Kontrola** porovná váš výsledek s očekávaným. Nic se nehodnotí.
- Rámečky **❓ Otázka**, **ℹ️ Poznámka**, **👉 Tip** a **⚠️ Pozor** doplňují výklad.

### Co budete potřebovat

| Co | Kde |
|---|---|
| Sára | rozbalený zip z kurzu (Windows nebo Linux) |
| Python, JupyterLab, NumPy, ipywidgets, Matplotlib a zvýraznění syntaxe gmacs | `pip install -r requirements.txt` ve složce s notebookem, před spuštěním JupyterLab |
| Krita 6 | [krita.org](https://krita.org/) |
| plugin `pga_filter` | `pga_filter-windows.zip` nebo `pga_filter-linux.zip` ze zipu kurzu |

Vedle notebooku nechte složku `lib` se soubory `sara.py`, `sara_notebook.py`, `sara_live.py` a `sara_params.py`, které notebook načítá, a s balíčkem pro zvýraznění gmacs, a složku `imgs` s obrázky.
""")

# --- 1. Sara ------------------------------------------------------------------
md(r"""
## 1. Spuštění Sáry a připojení

Notebook se Sárou komunikuje přes místní spojení (*agent door*). Ve výchozím stavu je vypnuté, Sáru proto spusťte jedním z těchto způsobů:

| Systém | Spuštění |
|---|---|
| Windows | dvojklik na `sara-with-door.cmd` |
| Linux | v terminálu `./sara.x86_64 -- --agent-door` |
| Sára už běží | v menu **Help → Connect an agent** |

Spusťte buňku níže. `sara.init()` se připojí k Sáře, otevře v ní nový dokument v sRGB ve výchozí velikosti (1920 × 1080), zaregistruje příkaz `%%gmacs` (sekce 4) a vrátí tři hodnoty: dokument `doc`, vrstvu vybranou v Sáře `layer` a její obsah jako pole čísel `pixels`.

> **⚠️ Pozor**
> Nový dokument nahradí ten, který máte v Sáře otevřený. Rozdělanou práci si nejdřív uložte.
""")

code("""
import sys
sys.path.insert(0, "lib")            # sara.py and its helpers live in lib/
import sara

doc, layer, pixels = sara.init()
""")

md(r"""
Buňka vypíše verzi Sáry, grafickou kartu a velikost plátna. Co umí `doc` a `layer`, shrnuje příloha na konci notebooku. V JupyterLab ukáže `Tab` za `doc.` seznam metod a `Shift + Tab` na názvu funkce její popis.

> **ℹ️ Poznámka**
> Chyba `No module named 'sara'` znamená, že vedle notebooku chybí složka `lib`. Jiná chyba při připojení obvykle znamená, že Sára neběží se zapnutým spojením.

> **❓ Otázka**
> Na jaké grafické kartě Sára počítá? Je to samostatná karta, nebo grafika v procesoru? Notebooky se dvěma kartami někdy použijí tu slabší.

### Testovací obrázek

Efekty budeme zkoušet na obrázku `imgs/0/color_chart.png`. `doc.import_image` ho otevře jako novou vrstvu pojmenovanou podle souboru, `color_chart`, a vybere ji:
""")

code("""
layer = doc.import_image("imgs/0/color_chart.png")   # a new layer called "color_chart"
pixels = layer.read()
layer.show()
""")

md(r"""
Nahoře je šest čistých barev (červená, zelená, modrá, žlutá, azurová a purpurová), uprostřed všechny odstíny s plnou sytostí a dole šedá od černé po bílou. Do vrstvy můžete v Sáře i malovat.

> **ℹ️ Poznámka**
> Opakované spuštění buňky načte obrázek do stávající vrstvy `color_chart`, druhou nepřidá. Stejně se chovají všechny vrstvy, které notebook pojmenuje.
""")

# --- 2. Arrays ----------------------------------------------------------------
md(r"""
## 2. Obrázek je pole čísel

Každý pixel jsou čtyři čísla: červená, zelená, modrá a průhlednost (alfa), zkráceně **RGBA**. Sára je ukládá jako desetinná čísla, 0 je nulová a 1 plná intenzita. V NumPy je obrázek pole tvaru **(výška, šířka, 4)**.

@img(image_as_array.png, 900, Mřížka pixelů a čtyři kanály jednoho pixelu)

> **⚠️ Pozor**
> NumPy indexuje **nejdřív řádkem `y`, pak sloupcem `x`**: `pixels[y, x]`. V kernelu je pořadí opačné, `(x, y)`, stejně jako v OpenGL.
""")

code("""
h, w = pixels.shape[:2]

print("shape:", pixels.shape)
print("dtype:", pixels.dtype)
print("min and max:", pixels.min(), pixels.max())
""")

code("""
y, x = 100, 200
print(f"pixel x = {x}, y = {y}:", pixels[y, x])
""")

md(r"""
Pixel leží v červeném poli, proto 1, 0, 0, 1: plná červená, žádná zelená ani modrá a plná alfa. Prázdný pixel by měl všechna čtyři čísla nulová, byl by černý a úplně průhledný.

### 🎯 Úkol 1

Kde na plátně je pixel `pixels[0, w - 1]`? A kde je `pixels[h - 1, 0]`? Odpovězte nejdřív sami, pak to ověřte výpisem. Barva pixelu prozradí, ve kterém poli leží.

<details><summary>💡 Nápověda</summary>

1. První index je řádek `y`, druhý sloupec `x`.
2. Řádek 0 je nahoře, sloupec 0 vlevo.
</details>

<details><summary>🔑 Řešení</summary>

`pixels[0, w - 1]` je pravý horní roh, purpurové pole s hodnotami 1, 0, 1, 1. `pixels[h - 1, 0]` je levý dolní roh, začátek šedé škály, černá s hodnotami 0, 0, 0, 1.

```python
print(pixels[0, w - 1], pixels[h - 1, 0])
```
</details>
""")

code("""
# Task 1: print both pixels

""")

md(r"""
### Z NumPy do Sáry

Pole spočítané v NumPy jde poslat do Sáry jako novou vrstvu. Tento obrázek vzniká jen ze souřadnic, červená roste zleva doprava a zelená shora dolů:
""")

code("""
import numpy as np

yy, xx = np.mgrid[0:h, 0:w]          # y and x of every pixel

gradient = np.zeros((h, w, 4), dtype=np.float32)
gradient[..., 0] = xx / (w - 1)      # red follows x
gradient[..., 1] = yy / (h - 1)      # green follows y
gradient[..., 3] = 1.0               # alpha, fully opaque

doc.new_layer("From numbers", gradient)
doc.show()
""")

md(r"""
`gradient[..., 0]` je kanál 0 (červená) všech pixelů najednou, tři tečky zastupují všechny řádky i sloupce. Takto NumPy pracuje s celým obrázkem bez cyklu.

Vrstva „From numbers“ zakrývá testovací obrázek, proto ji smažeme. Smazání je krok historie, `Ctrl + Z` v Sáře ho vrátí:
""")

code("""
doc.layer("From numbers").delete()
""")

md(r"""
### Jak pixely putují mezi Sárou a NumPy

Pixely vrstvy leží v paměti grafické karty, pole NumPy v paměti Pythonu. Přenos má dvě cesty:

@img(transport.png, 900, Příkazy jdou přes místní spojení, pixely přes dočasný soubor .npy)

- **Příkazy** jsou krátké zprávy v JSON, například „přečti vrstvu color_chart“. Na Linuxu jdou přes Unix socket, na Windows přes místní TCP spojení chráněné tokenem.
- **Pixely** jdou přes dočasný soubor `.npy`, vlastní formát NumPy: krátká hlavička s tvarem a typem pole, za ní surová data.

Při `layer.read()` Sára zkopíruje vrstvu z GPU do RAM, uloží ji jako `.npy` a odpoví cestou k souboru, který Python načte funkcí `np.load`. Zápis (`layer.write()`, `doc.new_layer()`) jde opačně přes `np.save` a v Sáře je jedním krokem historie.

> **ℹ️ Poznámka**
> Pole obsahuje přesně ta čísla, která Sára ukládá. Nic se neořezává, hodnoty mohou být i větší než 1 nebo záporné. Barvy jsou v dokumentu od `sara.init()` kódované v **sRGB** jako v běžném obrázku, co to znamená, ukáže sekce 5.

Na plátně 1920 × 1080 je jeden přenos $1920 \cdot 1080 \cdot 4 \text{ kanály} \cdot 4 \text{ B} \approx 33 \text{ MB}$, náhledy pod buňkami jsou proto zmenšené. Kolik přenos trvá na vašem počítači, změří tato buňka. Pomocnou vrstvu „Transfer test“ na konci smaže.
""")

code("""
import time

start = time.perf_counter()
pixels = layer.read()                          # GPU to NumPy
t_read = (time.perf_counter() - start) * 1000

scratch = doc.new_layer("Transfer test")
start = time.perf_counter()
scratch.write(pixels)                          # NumPy to GPU
t_write = (time.perf_counter() - start) * 1000
scratch.delete()                               # only the timing was needed

print(f"{pixels.nbytes / 1e6:.0f} MB, read {t_read:.0f} ms, write {t_write:.0f} ms")
""")

# --- 3. Why GPU ---------------------------------------------------------------
md(r"""
## 3. Proč GPU: jeden pixel, jedno vlákno

Inverze barev nahradí každý kanál $c$ hodnotou $1 - c$, z bílé udělá černou a z červené azurovou. Nejpřímočařeji ji napíše cyklus přes pixely v Pythonu, kvůli času jen na výřezu 200 × 200 px:
""")

code("""
crop = pixels[:200, :200]

def invert_loop(img):
    out = img.copy()
    rows, cols, _ = img.shape
    for y in range(rows):
        for x in range(cols):
            for c in range(3):                  # R, G, B, alpha stays
                out[y, x, c] = 1.0 - img[y, x, c]
    return out

start = time.perf_counter()
invert_loop(crop)
t_loop = time.perf_counter() - start

whole = t_loop * (h * w) / (200 * 200)
print(f"Python loop, 200 x 200 crop: {t_loop * 1000:.0f} ms")
print(f"estimate for the whole {w} x {h} canvas: {whole:.1f} s")
""")

md(r"""
Plynulý posuvník potřebuje aspoň 30 snímků za sekundu, tedy nejvýš **33 ms** na celý obrázek. Stejná inverze v NumPy, na celém plátně:
""")

code("""
def invert_numpy(img):
    out = img.copy()
    out[..., :3] = 1.0 - img[..., :3]    # every pixel, channels R, G, B
    return out

times = []
for _ in range(5):                       # the first run is often slower, keep the best of five
    start = time.perf_counter()
    invert_numpy(pixels)
    times.append(time.perf_counter() - start)
t_numpy = min(times)

print(f"NumPy, whole canvas: {t_numpy * 1000:.1f} ms")
print(f"about {whole / t_numpy:.0f} times faster than the loop")
""")

md(r"""
NumPy je rychlejší, protože cyklus přes pixely běží v předkompilovaném kódu v C. Inverze se do 33 ms nejspíš vejde. Efekt, který pro každý pixel čte desítky sousedů, bude ale úměrně pomalejší.

Grafická karta dá každému pixelu vlastní vlákno (*thread*) a tisíce vláken běží současně. Spouštějí se po skupinách (*workgroups*), v Sáře po 8 × 8 = 64 vláknech, takže se plátno rozdělí na dlaždice 8 × 8:

@img(thread_grid.png, 900, Plátno rozdělené na skupiny 8 × 8, každá má 64 vláken)

Pro plátno 1920 × 1080 px to je

$$\left\lceil \tfrac{1920}{8} \right\rceil \cdot \left\lceil \tfrac{1080}{8} \right\rceil = 240 \cdot 135 = 32\,400 \text{ skupin}, \qquad 32\,400 \cdot 64 = 2\,073\,600 \text{ vláken},$$

jedno vlákno na pixel. Kernel proto popisuje výpočet jednoho pixelu a GPU ho spustí pro všechny.
""")

# --- 4. First kernel ----------------------------------------------------------
md(r"""
## 4. První kernel

Kernel se píše do buňky, která začíná `%%gmacs`. Za ním je jméno vrstvy, ze které kernel čte, šipka a jméno vrstvy pro výsledek. Pokud výsledná vrstva neexistuje, Sára ji vytvoří.
""")

code("""
%%gmacs "color_chart" -> "Invert"
uniform float amount: hint_range(0, 1) = 1.0

def pixel(ivec2 at) -> vec4:
    vec4 c = src(at)
    return vec4(mix(c.rgb, 1.0 - c.rgb, amount), c.a)
""")

md(r"""
Pod buňkou se objeví posuvník `amount` a náhled nové vrstvy. Posuvník mění obrázek v notebooku i v Sáře.

| Řádek | Význam |
|---|---|
| `uniform float amount: hint_range(0, 1) = 1.0` | parametr kernelu s výchozí hodnotou 1.0. `hint_range(0, 1)` z něj udělá posuvník od 0 do 1, stejně jako v shaderech Godotu |
| `def pixel(ivec2 at) -> vec4:` | funkce, kterou GPU zavolá **pro každý pixel zvlášť**. `at` je pozice pixelu `(x, y)`, výsledek je jeho nová barva RGBA |
| `vec4 c = src(at)` | přečte pixel na pozici `at` ze zdrojové vrstvy (`color_chart`) |
| `c.rgb`, `c.a` | barevné kanály a alfa, stejný swizzle jako v GLSL |
| `mix(c.rgb, 1.0 - c.rgb, amount)` | přechod mezi původní a invertovanou barvou |

`mix` je lineární interpolace z GLSL. Pro `amount = 0` vrací původní obrázek, pro `amount = 1` inverzi:

$$\operatorname{mix}(a, b, t) = (1 - t)\,a + t\,b$$

@img(mix_graph.png, 480, Nová hodnota kanálu podle původní pro tři hodnoty amount)

> **❓ Otázka**
> Nastavte posuvník přesně na 0.5. Co se stane s obrázkem a proč?

<details><summary>🔑 Odpověď</summary>

$\operatorname{mix}(c, 1 - c, 0.5) = 0.5\,c + 0.5\,(1 - c) = 0.5$. Výsledek na `c` nezávisí, celý obrázek je stejně šedý.
</details>

### Jak dlouho kernel běží

Pod posuvníkem buňka ukazuje dva časy, například `GPU 0.21 ms, round trip 38 ms`:

| Čas | Co zahrnuje |
|---|---|
| GPU | jen běh kernelu na grafické kartě, změřený časovými značkami (*timestamps*) přímo na GPU |
| round trip | celou cestu z notebooku a zpět: příkaz přes spojení, běh kernelu a odpověď |

Výkon metody popisuje čas GPU. Round trip bývá o řád i víc delší, je to režie spojení a Pythonu a v hotovém filtru odpadne. Jedno měření kolísá, proto `sara.bench` spustí kernel, který naposledy zapsal danou vrstvu, víckrát po sobě a vrátí medián obou časů v milisekundách:
""")

code("""
t = sara.bench("Invert", runs=20)     # medians in ms: t.gpu_ms and t.round_trip_ms

print(f"Python loop (estimate): {whole * 1000:8.1f} ms")
print(f"NumPy:                  {t_numpy * 1000:8.1f} ms")
print(f"GPU kernel:             {t.gpu_ms:8.2f} ms")
print(f"round trip:             {t.round_trip_ms:8.1f} ms")
""")

# --- 5. Grey ------------------------------------------------------------------
md(r"""
## 5. Převod do šedi

Šedá barva má ve všech třech kanálech stejnou hodnotu. Jakou hodnotu z barvy spočítat, ukážeme dvěma způsoby.

### 🎯 Úkol 2: průměr kanálů

Nejjednodušší je průměr kanálů:

$$y = \frac{R + G + B}{3}$$

Buňka funguje, jen zatím vrací černou. Doplňte řádek s `TODO`. Posuvník `amount` přechází od barevného obrázku (0) k šedému (1).

<details><summary>💡 Nápověda</summary>

1. Jednotlivé kanály jsou `c.r`, `c.g` a `c.b`.
2. Nahraďte `0.0` na řádku `float y = 0.0` jejich průměrem.
</details>

<details><summary>🔑 Řešení</summary>

```python
float y = (c.r + c.g + c.b) / 3.0
```
</details>
""")

code("""
%%gmacs "color_chart" -> "Grey mean"
uniform float amount: hint_range(0, 1) = 1.0

def pixel(ivec2 at) -> vec4:
    vec4 c = src(at)
    float y = 0.0                 # TODO: mean of the three channels
    vec3 grey = vec3(y)
    return vec4(mix(c.rgb, grey, amount), c.a)
""")

md(r"""
### Proč průměr nestačí

Ve vrstvě „Grey mean“ dostane zelené i modré pole v horním pásu stejnou šedou, přestože zelená působí mnohem světleji. Průměr čisté červené, zelené i modré je vždy 1/3:

@img(grey_comparison.png, 900, Průměr dává stejnou šedou červené, zelené i modré, OKLab je rozliší)

Průměr dělá dvě chyby:

1. **Oko není stejně citlivé na všechny barvy.** Na zelenou je nejcitlivější, na modrou nejméně.
2. **Hodnoty nejsou úměrné množství světla.** Jsou kódované v sRGB, které dává víc přesnosti tmavým odstínům, kde oko rozliší nejvíc rozdílů. Hodnota 0.5 odpovídá zhruba 21 % světla (dekódováním vyjde 0.214).

Obojí řeší barevný prostor **OKLab**:

| Složka | Význam |
|---|---|
| $L$ | vnímaná světlost (*lightness*), 0 je černá, 1 bílá |
| $a$ | osa zelená ↔ červená |
| $b$ | osa modrá ↔ žlutá |

Šedá má $a = b = 0$. Převod do šedi tedy převede barvu do OKLabu, vynuluje $a$ a $b$, ponechá $L$ a převede barvu zpět. Pro `amount` mezi 0 a 1 se $a$ a $b$ jen zmenší.

Převod do OKLabu má čtyři kroky:

$$\text{sRGB} \xrightarrow{\text{dekódování}} \text{lineární RGB} \xrightarrow{M_1} \text{LMS} \xrightarrow{\sqrt[3]{\cdot}} \text{L'M'S'} \xrightarrow{M_2} \text{OKLab}$$

1. **Dekódování sRGB** převede každý kanál na množství světla:
$$c_\text{lin} = \begin{cases} \dfrac{c}{12.92} & c \le 0.04045 \\[6pt] \left(\dfrac{c + 0.055}{1.055}\right)^{2.4} & \text{jinak} \end{cases}$$
2. **Matice $M_1$** přepočítá lineární RGB na odezvu tří typů čípků v oku, LMS (*long, medium, short* podle vlnové délky). Toto L se světlostí nesouvisí.
3. **Třetí odmocnina** odpovídá tomu, že vnímaná světlost roste pomaleji než množství světla.
4. **Matice $M_2$** z odezvy čípků poskládá světlost $L$ a barevné osy $a$, $b$.

Zpět se jde stejnými kroky pozpátku. Odkud se vzaly matice a jak OKLab navazuje na CIE L\*a\*b\* z roku 1976, probere lekce o barevných prostorech. V kernelu jsou převody hotové v knihovnách Sáry.

### 🎯 Úkol 3: šedá podle OKLab

Buňka převede barvu do OKLabu a hned zpět, takže zatím vrací původní obrázek. Doplňte řádek s `TODO`, aby se $a$ a $b$ zmenšily podle `amount` a $L$ zůstalo.

<details><summary>💡 Nápověda</summary>

1. Řádky s `import` načtou knihovny Sáry: `linear_srgb_color_space` (dekódování a kódování sRGB) a `oklab_color_space` (převody mezi lineárním RGB a OKLabem).
2. Ve vektoru `lab` je $L$ složka `lab.x`, $a$ a $b$ jsou `lab.yz`. Swizzle funguje i pro zápis.
</details>

<details><summary>🔑 Řešení</summary>

```python
lab.yz = lab.yz * (1.0 - amount)
```
</details>
""")

code("""
%%gmacs "color_chart" -> "Grey OKLab"
import linear_srgb_color_space
import oklab_color_space

uniform float amount: hint_range(0, 1) = 1.0

def pixel(ivec2 at) -> vec4:
    vec4 c = src(at)
    vec3 lab = linear_srgb_to_oklab(srgb_to_linear_srgb(c.rgb))   # lab.x is L, lab.yz are a and b
    # TODO: scale a and b by (1 - amount), keep L
    vec3 rgb = linear_srgb_to_srgb(oklab_to_linear_srgb(lab))
    return vec4(rgb, c.a)
""")

md(r"""
Porovnejte „Grey mean“ a „Grey OKLab“ přepínáním viditelnosti, v Sáře ikonou oka v panelu vrstev, nebo z notebooku:
""")

code("""
grey_oklab = doc.layer("Grey OKLab")
grey_oklab.visible = False    # the layer below shows through
""")

md(r"""
Zpátky ji zapne `grey_oklab.visible = True`. V horním pásu se vrstvy nejvíc liší u žluté, zelené a modré. V prostředním pásu průměr střídá stejně tmavé a stejně světlé pruhy, kdežto OKLab ukáže žlutou jako nejsvětlejší a modrou jako nejtmavší.
""")

# --- 6. gmacs is GLSL ---------------------------------------------------------
md(r"""
## 6. gmacs: dialekt GLSL

Jazyk kernelů se jmenuje **gmacs**. Je to dialekt GLSL: výrazy, typy (`vec4`, `ivec2`), swizzle (`c.rgb`) i vestavěné funkce (`mix`, `step`, `dot`, …) má stejné jako GLSL z OpenGL. Bloky ale zapisuje jako Python a několik jmen má jinak.

### Fragment shader a compute kernel

Fragment shader z OpenGL dostane jeden fragment vzniklý rasterizací trojúhelníku a vrátí jeho barvu. Compute kernel trojúhelníky nemá. Spustí se mřížka vláken, každé vlákno zná svou pozici a samo čte a zapisuje obrázky.

@img(fragment_vs_compute.png, 900, OpenGL pipeline a compute kernel)

Funkce `pixel(ivec2 at) -> vec4` se fragment shaderu podobá, dostane pozici a vrátí barvu. `src()` ale přečte **libovolný** pixel vrstvy, nejen ten na pozici `at`. To později využije rozostření i detekce hran.

### Rozdíly proti GLSL

| GLSL | gmacs |
|---|---|
| `{ … }` a středníky | odsazení o 4 mezery, dvojtečka na konci řádku |
| `float f(vec3 c) { … }` | `def f(vec3 c) -> float:` |
| `&&`, `\|\|`, `!` | `and`, `or`, `not` |
| `for (int i = 0; i < n; i++)` | `for i in range(0, n):` |
| `switch` / `case` / `default` | `match` / `case` / `case _` |
| `imageLoad`, `imageStore`, `imageSize` | `image_load`, `image_store`, `image_size` |
| `ivec2(gl_GlobalInvocationID.xy)` | `pixel` |
| `#include` | `import` |
| `// komentář` | `# komentář` |

Následuje inverze ze sekce 4 a pod ní zhruba GLSL 450, které z ní vznikne, přeložené ručně a zjednodušené.
""")

md(r"""
**gmacs, jak jste ho napsali:**

```python
uniform float amount: hint_range(0, 1) = 1.0

def pixel(ivec2 at) -> vec4:
    vec4 c = src(at)
    return vec4(mix(c.rgb, 1.0 - c.rgb, amount), c.a)
```
""")

md(r"""
**GLSL 450 po překladu (zjednodušeno):**

```glsl
#version 450
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;   // 8 x 8 workgroup

layout(rgba32f, binding = 0) uniform readonly image2D sa_source;     // color_chart
layout(rgba32f, binding = 1) uniform image2D sa_target;              // Invert

layout(push_constant, std430) uniform Params {   // slider values
    vec2 origin;      // top left corner of the processed area
    float amount;     // your uniform
    float _pad_0;     // padding to 16 bytes, added by the compiler
};

vec4 src(ivec2 at) {                             // reads the source layer
    ivec2 p = at - ivec2(origin);
    ivec2 size = imageSize(sa_source);
    if (p.x < 0 || p.y < 0 || p.x >= size.x || p.y >= size.y)
        return vec4(0.0);                        // outside the layer, transparent black
    return imageLoad(sa_source, p);
}

vec4 pixel_function(ivec2 at) {                  // your pixel function
    vec4 c = src(at);
    return vec4(mix(c.rgb, 1.0 - c.rgb, amount), c.a);
}

void main() {                                    // runs once per thread
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (pixel.x >= imageSize(sa_target).x || pixel.y >= imageSize(sa_target).y)
        return;                                  // threads outside the image do nothing
    imageStore(sa_target, pixel, pixel_function(pixel + ivec2(origin)));
}
```

Deklarace obrázků, blok parametrů (*push constants*), `src` a `main` doplní Sára, v buňce zůstává jen výpočet jednoho pixelu. Formát `rgba32f` odpovídá 32bitovému dokumentu, u 16bitového je to `rgba16f`.

<details><summary>Celý kernel v gmacs, který Sára kolem vaší funkce vygeneruje</summary>

```python
kernel invert

image sa_source: readonly
image sa_target

params:
    vec2 origin
    float amount


def src(ivec2 at) -> vec4:
    ivec2 at_source = at - ivec2(origin)
    ivec2 source_size = image_size(sa_source)
    if at_source.x < 0 or at_source.y < 0 or at_source.x >= source_size.x or at_source.y >= source_size.y:
        return vec4(0.0)
    return image_load(sa_source, at_source)


def luminance(vec3 rgb) -> float:
    return dot(rgb, vec3(0.2126, 0.7152, 0.0722))


import sara_pixels/invert    # your cell, with pixel renamed to pixel_function


def invert():
    ivec2 target_size = image_size(sa_target)
    if pixel.x >= target_size.x or pixel.y >= target_size.y:
        return
    image_store(sa_target, pixel, pixel_function(pixel + ivec2(origin)))
```

Funkce `luminance` je vážený součet kanálů podle normy Rec. 709. Váhy odpovídají citlivosti oka, ale počítá s hodnotami kódovanými v sRGB, takže světlost vychází jen přibližně. Proto jsme v úkolu 3 použili OKLab.
</details>
""")

md(r"""
### Cesta jedné buňky

@img(cell_journey.png, 900, Buňka, obalení, gmacs na GLSL, SPIR-V, GPU, nová vrstva, náhled)

Překlad proběhne jen při spuštění buňky. Pohyb posuvníkem pošle Sáře jen nové hodnoty parametrů a kernel se spustí znovu bez překladu.

### Krátký tvar a celý kernel

Krátký tvar s funkcí `pixel` stačí, když jedna vrstva vstupuje, jedna vystupuje a barva pixelu závisí na něm a na jeho sousedech. `src()` mimo vrstvu vrací průhlednou černou, okraje proto hlídat nemusíte. V buňce můžete psát pomocné funkce, cykly, podmínky a struktury a načítat knihovny přes `import`.

Do buňky jde napsat i celý kernel, jako ten v rozbalovacím bloku výše. Umí vše, co compute shader v GLSL 450, třeba sdílenou paměť skupiny (*shared memory*), `barrier()` a atomické operace. Parametry (`uniform`) se předávají jako push constants, dohromady nejvýš 128 bajtů, což stačí na desítky posuvníků.
""")

# --- 7. NumPy -----------------------------------------------------------------
md(r"""
## 7. Stejná matematika v NumPy

Krita kernely spouštět neumí, její pluginy běží v Pythonu na procesoru. Proto každý efekt přepíšeme do NumPy:

| Kernel na GPU | NumPy na CPU |
|---|---|
| popisuje, co se stane s **jedním** pixelem | popisuje, co se stane s **celým polem** najednou |
| `c.rgb` je barva jednoho pixelu | `img[..., :3]` jsou barvy všech pixelů, tvar (výška, šířka, 3) |
| `mix(a, b, t)` je vestavěná funkce | `mix` si napíšeme sami, je to jeden řádek |

Efekt pro Kritu má dvě části s pevnými jmény:

- **`PARAMS`**: ke jménu každého parametru trojice (výchozí hodnota, minimum, maximum),
- **`apply(img, ...)`**: dostane pole tvaru (výška, šířka, 4) a parametry pojmenované podle `PARAMS` a vrátí pole stejného tvaru.
""")

code("""
def mix(a, b, t):
    \"\"\"Same as mix in GLSL: (1 - t) * a + t * b.\"\"\"
    return (1.0 - t) * a + t * b


PARAMS = {
    "amount": (1.0, 0.0, 1.0),   # (default, minimum, maximum)
}


def apply(img, amount):
    out = img.copy()
    rgb = img[..., :3]
    out[..., :3] = mix(rgb, 1.0 - rgb, amount)
    return out
""")

md(r"""
Místo barvy jednoho pixelu `c.rgb` je tu pole barev všech pixelů `rgb`, zbytek je stejný. `sara.live` vytvoří z `PARAMS` posuvníky, při každém pohybu zavolá `apply` na vrstvu přečtenou na začátku a výsledek zapíše do vrstvy `target`:
""")

code("""
sara.live(apply, PARAMS, target="Invert NumPy")
""")

md(r"""
Posuvník reaguje pomaleji než u kernelu, protože se celý obrázek pokaždé spočítá na procesoru a přenese do Sáry.

### ✅ Kontrola

`sara.check` porovná vrstvu spočítanou na GPU s polem z NumPy a vypíše největší rozdíl.

> **⚠️ Pozor**
> Posuvník kernelu „Invert“ v sekci 4 nejdřív vraťte na 1. `sara.check` porovnává s tím, co je ve vrstvě právě teď.
""")

code("""
sara.check("Invert", apply(pixels, amount=1.0))
""")

md(r"""
Rozdíl v řádu tisícin je v pořádku, GPU a CPU zaokrouhlují trochu jinak a dokument může mít menší bitovou hloubku. Rozdíl v desetinách znamená, že obě verze počítají něco jiného.

### 🎯 Úkol 4: průměr v NumPy

Přepište průměr z úkolu 2 do NumPy. Buňka funguje, jen zatím vrací černou.

<details><summary>💡 Nápověda</summary>

1. `rgb` má tvar (výška, šířka, 3), průměr potřebujete přes poslední osu, tedy přes kanály.
2. Metoda `mean` bere parametr `axis`, osu, přes kterou se průměruje. Poslední osa je `-1`.
</details>

<details><summary>🔑 Řešení</summary>

```python
y = rgb.mean(axis=-1)
```
</details>
""")

code("""
PARAMS_GREY = {
    "amount": (1.0, 0.0, 1.0),
}


def apply_grey_mean(img, amount):
    out = img.copy()
    rgb = img[..., :3]
    y = np.zeros(img.shape[:2], dtype=np.float32)    # TODO: mean of R, G, B for every pixel
    grey = np.stack([y, y, y], axis=-1)              # (h, w) to (h, w, 3)
    out[..., :3] = mix(rgb, grey, amount)
    return out


sara.live(apply_grey_mean, PARAMS_GREY, target="Grey mean NumPy")
""")

md(r"""
### ✅ Kontrola

Porovnání s kernelem „Grey mean“ z úkolu 2, jeho posuvník nastavte na 1:
""")

code("""
sara.check("Grey mean", apply_grey_mean(pixels, amount=1.0))
""")

md(r"""
### OKLab v NumPy

V NumPy převody napíšeme sami, jsou to čtyři kroky ze sekce 5. Buňku stačí spustit.

`rgb @ M1.T` vynásobí maticí $M_1$ barvu každého pixelu najednou. Transpozice je tu proto, že barva pixelu je v NumPy řádkový vektor. Inverzní matice spočítá `np.linalg.inv`.
""")

code("""
def srgb_to_linear(c):
    \"\"\"sRGB decoding, mirrored through zero so that negative values survive.\"\"\"
    a = np.abs(c)
    linear = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
    return np.sign(c) * linear


def linear_to_srgb(c):
    \"\"\"sRGB encoding, the inverse of srgb_to_linear.\"\"\"
    a = np.abs(c)
    encoded = np.where(a <= 0.0031308, a * 12.92, 1.055 * a ** (1 / 2.4) - 0.055)
    return np.sign(c) * encoded


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
    lms = rgb @ M1.T                 # M1 times the colour of every pixel
    return np.cbrt(lms) @ M2.T


def oklab_to_linear(lab):
    lms = (lab @ M2_INV.T) ** 3
    return lms @ M1_INV.T
""")

md(r"""
### 🎯 Úkol 5: šedá podle OKLab v NumPy

Jako v úkolu 3 buňka převede barvy do OKLabu a zpět a zatím vrací původní obrázek. Doplňte řádek s `TODO`.

<details><summary>💡 Nápověda</summary>

1. `lab` má tvar (výška, šířka, 3). $L$ je `lab[..., 0]`, $a$ a $b$ jsou `lab[..., 1:]`.
2. Výřez pole jde násobit na místě, operátorem `*=`.
</details>

<details><summary>🔑 Řešení</summary>

```python
lab[..., 1:] *= 1.0 - amount
```
</details>
""")

code("""
def apply_grey_oklab(img, amount):
    out = img.copy()
    lab = linear_to_oklab(srgb_to_linear(img[..., :3]))    # lab[..., 0] is L, lab[..., 1:] are a and b
    # TODO: scale a and b by (1 - amount), keep L
    out[..., :3] = linear_to_srgb(oklab_to_linear(lab))
    return out


sara.live(apply_grey_oklab, PARAMS_GREY, target="Grey OKLab NumPy")
""")

md(r"""
### ✅ Kontrola

Porovnání s kernelem „Grey OKLab“ z úkolu 3, posuvník na 1:
""")

code("""
sara.check("Grey OKLab", apply_grey_oklab(pixels, amount=1.0))
""")

md(r"""
### Rychlost: NumPy a GPU

Převod do OKLabu dá procesoru víc práce než inverze:
""")

code("""
times = []
for _ in range(3):
    start = time.perf_counter()
    apply_grey_oklab(pixels, amount=1.0)
    times.append(time.perf_counter() - start)

t = sara.bench("Grey OKLab", runs=20)
print(f"NumPy:      {min(times) * 1000:8.1f} ms")
print(f"GPU kernel: {t.gpu_ms:8.2f} ms")
""")

md(r"""
> **❓ Otázka**
> Porovnejte oba časy s inverzí ze sekcí 3 a 4. Kolikrát se zpomalilo NumPy a kolikrát kernel?

<details><summary>🔑 Odpověď</summary>

NumPy se zpomalí výrazně. Každý pixel prochází dekódováním, dvěma maticemi, třetí odmocninou a zpět, a každý krok vytvoří v paměti nové pole velikosti celého obrázku. Kernel se zpomalí mnohem méně. Výpočtů na pixel je pořád málo a většinu času zabere čtení a zápis vrstvy v paměti grafické karty.
</details>
""")

# --- 8. Krita -----------------------------------------------------------------
md(r"""
## 8. Do Krity: šablona pluginu

Šablona `pga_filter` obstará okno, čtení vrstvy i zápis výsledku. Vy do ní vložíte jen `PARAMS` a `apply`.

### Instalace (jednou)

1. V Kritě zvolte **Tools → Scripts → Import Python Plugin from File…** a vyberte zip pro svůj systém.
2. Restartujte Kritu.
3. V **Settings → Configure Krita… → Python Plugin Manager** zaškrtněte **PGA filter** a Kritu restartujte ještě jednou.

### Co je ve složce

Plugin leží ve složce s daty Krity, kterou otevře **Settings → Manage Resources… → Open Resource Folder** (na Windows obvykle `%APPDATA%\krita`, na Linuxu `~/.local/share/krita`):

```
pykrita/
├── pga_filter.desktop
└── pga_filter/
    ├── effect.py
    ├── Manual.html
    ├── dialog.py, engine.py, pixels.py, ...
    └── vendor/
```

| Soubor | K čemu je |
|---|---|
| `pga_filter.desktop` | popis pluginu, podle kterého ho Krita najde |
| `effect.py` | **váš kód**: `TITLE`, `PARAMS` a `apply` |
| `Manual.html` | návod, Krita ho ukáže v Python Plugin Manageru |
| ostatní `.py` | okno s ovládacími prvky, náhled, čtení a zápis vrstvy. Upravovat je nemusíte |
| `vendor/` | přibalený NumPy, Krita vlastní nemá |

Po instalaci obsahuje `effect.py` inverzi ze sekce 7:

```python
import numpy as np

TITLE = "Invert"


def mix(a, b, t):
    return (1.0 - t) * a + t * b


PARAMS = {
    "amount": (1.0, 0.0, 1.0),
}


def apply(img, amount):
    out = img.copy()
    rgb = img[..., :3]
    out[..., :3] = mix(rgb, 1.0 - rgb, amount)
    return out
```

### Spuštění

1. Otevřete v Kritě obrázek, třeba fotku, a vyberte vrstvu.
2. **Tools → Scripts → Invert…** (podle `TITLE`) otevře okno s posuvníky z `PARAMS` a s náhledem.
3. Klikněte na **Apply**. Je-li v obrázku výběr, filtr se použije jen na něj. Výsledek nahradí vrstvu její kopií, původní vrstvu vrátí dvakrát `Ctrl + Z`.

Šablona za vás řeší, v čem se Krita od notebooku liší: vrstvy v 8, 16 i 32 bitech a jiné pořadí kanálů (8bitové barvy Krita ukládá jako BGRA). `apply` proto vždy dostane pole RGBA s hodnotami 0 až 1 jako v notebooku, u běžného obrázku kódované v sRGB.

> **👉 Tip**
> Po uložení `effect.py` stačí okno filtru zavřít a znovu otevřít, Kritu restartovat nemusíte.

### 🎯 Úkol 6: šedá podle OKLab v Kritě

Vložte do `effect.py` verzi s OKLabem z úkolu 5 a vyzkoušejte ji na fotce.

<details><summary>💡 Nápověda</summary>

1. Z notebooku zkopírujte buňku s převody (`srgb_to_linear` až `oklab_to_linear`, včetně matic) a vložte ji pod `mix`.
2. Místo stávajících `PARAMS` a `apply` vložte `PARAMS_GREY` a `apply_grey_oklab` a přejmenujte je na `PARAMS` a `apply`. Jiná jména šablona nenajde. `TITLE` přepište na "Grey OKLab".
</details>

<details><summary>🔑 Řešení</summary>

Pod `mix` a převody z notebooku stojí:

```python
TITLE = "Grey OKLab"

PARAMS = {
    "amount": (1.0, 0.0, 1.0),
}


def apply(img, amount):
    out = img.copy()
    lab = linear_to_oklab(srgb_to_linear(img[..., :3]))
    lab[..., 1:] *= 1.0 - amount
    out[..., :3] = linear_to_srgb(oklab_to_linear(lab))
    return out
```
</details>
""")

# --- Summary ------------------------------------------------------------------
md(r"""
## Shrnutí

- Obrázek je pole tvaru **(výška, šířka, 4)**. V NumPy se indexuje `pixels[y, x]`, v kernelu `(x, y)`.
- Příkazy jdou mezi notebookem a Sárou přes místní spojení, pixely přes dočasný soubor `.npy`.
- Cyklus přes pixely v Pythonu je pomalý. NumPy počítá s celým polem najednou, GPU dává každému pixelu vlastní vlákno.
- Kernel v buňce `%%gmacs` je funkce `pixel(ivec2 at) -> vec4`. `src(at)` čte zdrojovou vrstvu, `uniform` s `hint_range` je posuvník. Jeho výkon popisuje čas GPU z `sara.bench`.
- **gmacs je dialekt GLSL** s odsazením místo závorek. Zbytek kernelu kolem vaší funkce doplní Sára.
- Průměr kanálů není dobrá šedá. Hodnoty v sRGB je potřeba dekódovat a světlost $L$ z OKLabu odpovídá tomu, jak světlost vnímá oko.
- Efekt pro Kritu je **`PARAMS` a `apply(img, ...)`** v NumPy, šablona `pga_filter` ho spustí beze změny.

### Co jsme vynechali

- **Barevné prostory podrobně**: odkud pocházejí matice OKLabu, CIE L\*a\*b\*, proč sRGB kóduje právě takto. To probere lekce o barevných prostorech.
- **Sousední pixely.** Dnešní efekty byly bodové operace, výsledek pixelu závisel jen na něm samém. Rozostření, hrany a sdílená paměť skupiny přijdou později.
- **Hodnoty nad 1.** Sára nic neořezává, takže kanál může být větší než 1 a inverze z něj udělá záporné číslo. K tomu se vrátíme u HDR.
- **Jak šablona čte pixely z Krity.** Podrobnosti jsou v okomentovaném `pga_filter/pixels.py`.

### Bonusové úkoly

1. **Pruhy.** Funkce `pixel` dostává pozici `at`, efekt tedy může záviset na tom, kde pixel leží. Upravte buňku níže tak, aby invertovala jen každý druhý svislý pruh. Pak zkuste pruhy úhlopříčně, stačí místo `at.x` použít jiný výraz z `at.x` a `at.y`.
2. **Práh.** Kernel, který obrázek převede jen na černou a bílou podle světlosti $L$ z OKLabu a posuvníku `level`. Hodí se funkce `step(edge, x)` z GLSL, která vrací 0 pro `x < edge`, jinak 1.
3. **Prohození kanálů.** Co udělá `vec4(c.brg, c.a)`? A `c.gbr`? Odhadněte výsledek a teprve pak ho spusťte.

<details><summary>💡 Nápověda k pruhům</summary>

1. `at.x / int(width)` je celočíselné dělení, číslo pruhu, ve kterém pixel leží (0, 1, 2, …).
2. Lichý pruh poznáte podle zbytku po dělení dvěma: `stripe % 2 == 1`.
3. Podmínka se v gmacs píše jako v Pythonu, bez závorek a s dvojtečkou.
</details>

<details><summary>🔑 Řešení pruhů</summary>

```python
if stripe % 2 == 1:
    return vec4(1.0 - c.rgb, c.a)
```

Úhlopříčné pruhy dá například `int stripe = (at.x + at.y) / int(width)`.
</details>
""")

code("""
%%gmacs "color_chart" -> "Stripes"
uniform float width: hint_range(2, 64) = 16.0

def pixel(ivec2 at) -> vec4:
    vec4 c = src(at)
    int stripe = at.x / int(width)
    # TODO: return odd stripes inverted
    return c
""")

md(r"""
### Nápady na semestrální práci

- Plugin s několika bodovými efekty v jednom okně (inverze, šedá, práh, prohození kanálů). Mezi efekty se přepíná rozbalovacím seznamem, v `PARAMS` je to seznam textů jako `["Invert", "Grey", "Threshold"]`.
- **Duotone**: obrázek převedený na přechod mezi dvěma barvami podle světlosti $L$. Obě barvy si uživatel vybere, v `PARAMS` jsou to dvě barvy jako `"#1a2b6d"`. Přechod je v OKLabu rovnoměrnější než v sRGB, víc v lekci o barevných prostorech.
""")

# --- Appendix: the API --------------------------------------------------------
md(r"""
## Příloha: přehled funkcí

Funkce, které notebook používá, jsou v souborech `sara.py`, `sara_notebook.py` a `sara_live.py` ve složce `lib`. Podrobný popis každé ukáže `help(...)`, například `help(doc.new_layer)`.

**Modul `sara`**

| Volání | Co dělá |
|---|---|
| `sara.init()` | připojí se k Sáře, zaregistruje `%%gmacs` a vrátí `doc, layer, pixels` |
| `sara.connect()` | jen se připojí a vrátí `doc` |
| `sara.live(apply, PARAMS, target="...")` | posuvníky pro NumPy funkci, výsledek zapisuje do vrstvy `target` |
| `sara.check("vrstva", pole)` | vypíše největší rozdíl mezi vrstvou a polem |
| `sara.bench("vrstva", runs=20)` | medián času kernelu, který vrstvu naposledy zapsal: `.gpu_ms` a `.round_trip_ms` |

**Dokument `doc`**

| Volání | Co dělá |
|---|---|
| `doc.layer("color_chart")`, `doc.layer(0)` | vrstva podle jména, nebo podle pořadí odspodu (0 je nejspodnější) |
| `doc.layers()` | seznam vrstev odspodu: jméno, viditelnost, krytí a která je vybraná |
| `doc.new_layer("jméno", pole)` | nová vrstva nad aktivní. Pole je nepovinné, bez něj je vrstva prázdná. Vrstvu, která se tak už jmenuje, použije znovu |
| `doc.import_image("cesta.png")` | otevře obrázek jako vrstvu pojmenovanou podle souboru |
| `doc.show()` | zmenšený náhled složeného obrazu ze všech vrstev |
| `doc.undo(steps=1)` | vrátí kroky historie |
| `doc.undo_step("jméno")` | pojmenuje další krok historie |
| `doc.close()` | ukončí spojení, Sára běží dál |

**Vrstva `layer`**

| Volání | Co dělá |
|---|---|
| `layer.read()` | pixely celé vrstvy jako pole `float32` tvaru (výška, šířka, 4) |
| `layer.read([x, y, w, h])` | jen obdélník |
| `layer.write(pole)` | zapíše pole do vrstvy jako jeden krok historie |
| `layer.show()` | zmenšený náhled vrstvy |
| `layer.visible = False` | skryje vrstvu, `True` ji zase zobrazí |
| `layer.opacity = 0.5` | krytí vrstvy od 0 do 1 |
| `layer.duplicate()` | kopie vrstvy nad ní, vrátí novou vrstvu |
| `layer.select()` | vybere vrstvu v Sáře |
| `layer.delete()` | smaže vrstvu jako krok historie, `doc.undo()` ji vrátí |

**Buňka `%%gmacs`**

| Zápis | Co dělá |
|---|---|
| `%%gmacs "Zdroj" -> "Cíl"` | spustí kernel nad vrstvou „Zdroj“ a zapíše výsledek do vrstvy „Cíl“, kterou vytvoří, pokud neexistuje |
| `uniform float x: hint_range(0, 1) = 0.5` | parametr kernelu s posuvníkem od 0 do 1. Třetí číslo nastaví krok: `hint_range(0, 1, 0.1)` |
| `def pixel(ivec2 at) -> vec4:` | funkce pro jeden pixel |
| `src(at)` | barva zdrojové vrstvy na pozici `at`, mimo vrstvu průhledná černá |
| `import oklab_color_space` | načte knihovnu Sáry |
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
for i, cell in enumerate(nb["cells"]):
    cell["id"] = f"l0-{i:02d}"
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(OUT, len(cells), "cells")
