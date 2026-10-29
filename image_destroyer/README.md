# Image Destroyer

`image_destroyer` turns an input image into a reproducible synthetic damaged
image. It leaves the original image unchanged and returns an RGB image with
destroyed pixels filled in black.

## Quick start

From the `UnsupervisedLearning` repository root, use the package's default
Perlin-noise strategy:

```python
from PIL import Image

from image_destroyer import ImageDestroyer

with Image.open("input.png") as image:
    damaged = ImageDestroyer(image, seed=42).destroy()

damaged.save("output.png")
```

By default, the strategy uses noise scale `0.01` and threshold `0.91`. A
size-7 binary closing filter connects nearby destroyed regions. Set `seed` to
an integer to reproduce the same damage mask for the same image dimensions.

## Inspect multiple results

To generate 100 variations and a contact sheet for manual inspection, run this
from the repository root:

```powershell
uv run python -m image_destroyer.preview_image_destruction path\to\image.png
```

The script saves the source image, 100 full-size PNG variations, and a
`contact-sheet.png` in a timestamped subdirectory of
`image_destroyer_previews/`. It starts with seed `42` and increments the seed
for each variation. For example, use the repository's sample image:

```powershell
uv run python -m image_destroyer.preview_image_destruction super_resolution/sr_test.png
```

Use `--count`, `--seed`, `--columns`, or `--output-dir` to customize the run.
The preview output directory is ignored by Git.

## Choose a strategy

To choose a different strategy, pass it when creating the destroyer:

```python
from PIL import Image

from image_destroyer import ImageDestroyer, PropagationStrategy

with Image.open("input.png") as image:
    damaged = ImageDestroyer(
        image, strategy=PropagationStrategy(), seed=42
    ).destroy()

damaged.save("output.png")
```

Other options include `CellularAutomataStrategy()` and a configured
`PerlinNoiseStrategy(PerlinNoiseConfig(...))`.

## Customize mask processing

The default size-7 closing filter can be replaced by explicitly supplying
filters to `destroy`. An empty iterable disables filtering:

```python
from PIL import Image

from image_destroyer import BinaryClosing, ImageDestroyer

with Image.open("input.png") as image:
    result = ImageDestroyer(image, seed=42).destroy(
        filters=[BinaryClosing(size=3)]
    )
    unfiltered = ImageDestroyer(image, seed=42).destroy(filters=[])
```

You can override the strategy for an individual call too:

```python
from PIL import Image

from image_destroyer import ImageDestroyer, PropagationStrategy

with Image.open("input.png") as image:
    result = ImageDestroyer(image, seed=42).destroy(
        strategy=PropagationStrategy()
    )
```

If filters are passed explicitly, they replace the default closing filter;
pass `BinaryClosing(size=7)` to keep the default connection radius alongside
other custom filters.
