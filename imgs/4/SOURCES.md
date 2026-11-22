# Where the photos come from

Four photos here are not drawn by `build/lesson_04_diagrams.py`. Two are
sample pictures that [scikit-image](https://scikit-image.org) ships in
`skimage.data`, copied unchanged except that `brick.png` is stored as RGB.
`lenna.png` is the classic test image, copied unchanged from the USC-SIPI
image database's 512 × 512 version.

| File | Picture | Licence |
|---|---|---|
| `astronaut.png` | Eileen Collins, NASA astronaut, from the NASA Great Images database (`skimage.data.astronaut`) | public domain, no known copyright restrictions |
| `brick.png` | Bricks25 from CC0Textures (`skimage.data.brick`) | CC0 |
| `car.png` | A road photo from a figure in Marcin Iwanowski's publication on ResearchGate (publication 220872176, figure 3), https://www.researchgate.net/profile/Marcin-Iwanowski/publication/220872176/figure/fig3/AS:667827040161795@1536233782147/The-camera-mounted-over-the-road-and-looking-at-the-incoming-traffic-a-the-view-from.jpg. The course's author replaced the front car's plate with "PGA 321", masked the other plates, scaled it to 512 pixels tall and cropped it to 512 × 512 | not stated by the source, used for teaching |
| `lenna.png` | Lena Forsén, a scan of the Playboy centrefold of November 1972, the USC-SIPI test image | copyright Playboy Enterprises, not freely licensed, used here as the traditional test image |

`astronaut_blur.png` is `astronaut.png` blurred by the lesson's own
`filter_fft`, and the diagram script writes it again on every run.

`car_blur.png` is `car.png` blurred by the same `filter_fft`, and
`car_motion.png` is `car.png` blurred along a line with the lesson's
`motion_filter`, with shot and read noise added and rounded to eight bits. The
diagram script writes both again on every run. Its other pictures are drawn by
the same script.
