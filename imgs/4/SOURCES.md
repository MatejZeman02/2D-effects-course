# Where the photos come from

Two photos here are not drawn by `build/lesson_04_diagrams.py`. Both are
sample pictures that [scikit-image](https://scikit-image.org) ships in
`skimage.data`, copied unchanged except that `brick.png` is stored as RGB.

| File | Picture | Licence |
|---|---|---|
| `astronaut.png` | Eileen Collins, NASA astronaut, from the NASA Great Images database (`skimage.data.astronaut`) | public domain, no known copyright restrictions |
| `brick.png` | Bricks25 from CC0Textures (`skimage.data.brick`) | CC0 |

`astronaut_blur.png` is `astronaut.png` blurred by the lesson's own
`blur_fft`, and the diagram script writes it again on every run.
