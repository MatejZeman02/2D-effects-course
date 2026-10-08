# Where the pictures come from

`rocket_hdr.npz` is made from the rocket photo that
[scikit-image](https://scikit-image.org) ships as `skimage.data.rocket`, a
Falcon 9 on its launch pad photographed by SpaceX and released into the public
domain. `build/lesson_06_diagrams.py` converts it to linear light and makes
everything near white up to a hundred times brighter, so the pad's lamps are
lights again rather than clipped white. It is stored as half floats under the
key `rgb`, which is what a Sara layer keeps, and the script writes it again on
every run.

| File | Picture | Licence |
|---|---|---|
| `rocket_hdr.npz` | Falcon 9 on the pad, SpaceX (`skimage.data.rocket`), made HDR as above | public domain |

Every other picture here is drawn by the diagram script.
