# Where the photos come from

Three photos here are not drawn by `build/lesson_04_diagrams.py`. Two are
sample pictures that [scikit-image](https://scikit-image.org) ships in
`skimage.data`, copied unchanged except that `brick.png` is stored as RGB.
`lenna.png` is the classic test image, copied unchanged from the USC-SIPI
image database's 512 × 512 version.

| File | Picture | Licence |
|---|---|---|
| `astronaut.png` | Eileen Collins, NASA astronaut, from the NASA Great Images database (`skimage.data.astronaut`) | public domain, no known copyright restrictions |
| `brick.png` | Bricks25 from CC0Textures (`skimage.data.brick`) | CC0 |
| `lenna.png` | Lena Forsén, a scan of the Playboy centrefold of November 1972, the USC-SIPI test image | copyright Playboy Enterprises, not freely licensed, used here as the traditional test image |

`astronaut_blur.png` is `astronaut.png` blurred by the lesson's own
`filter_fft`, and the diagram script writes it again on every run.

`text_motion.png` is synthetic, not a real photo. The diagram script sets
a few lines of text, lights them with a vignette, blurs them along a line with
the lesson's `motion_filter`, adds shot and read noise and rounds to eight bits.
Its other pictures are drawn by the same script.
