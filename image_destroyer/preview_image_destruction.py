from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw

from image_destroyer import ImageDestroyer


def create_previews(
    input_path: Path,
    output_root: Path,
    *,
    count: int = 100,
    seed: int = 42,
    columns: int = 10,
) -> Path:
    if count < 1:
        raise ValueError("count must be at least 1")
    if columns < 1:
        raise ValueError("columns must be at least 1")

    with Image.open(input_path) as image:
        source = image.convert("RGB")

    run_dir = output_root / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    source.save(run_dir / "source.png")

    cell_width = 180
    cell_height = 204
    thumbnail_height = cell_height - 24
    rows = (count + columns - 1) // columns
    contact_sheet = Image.new(
        "RGB",
        (columns * cell_width, rows * cell_height),
        (245, 245, 245),
    )
    draw = ImageDraw.Draw(contact_sheet)

    for index in range(count):
        variant_seed = seed + index
        variant = ImageDestroyer(source, seed=variant_seed).destroy()
        variant.save(
            run_dir / f"variant-{index + 1:03d}-seed-{variant_seed}.png"
        )

        thumbnail = variant.copy()
        thumbnail.thumbnail(
            (cell_width, thumbnail_height),
            Image.Resampling.LANCZOS,
        )
        column, row = index % columns, index // columns
        left, top = column * cell_width, row * cell_height
        contact_sheet.paste(
            thumbnail,
            (
                left + (cell_width - thumbnail.width) // 2,
                top + (thumbnail_height - thumbnail.height) // 2,
            ),
        )
        draw.text(
            (left + 6, top + thumbnail_height + 4),
            f"Seed {variant_seed}",
            fill=(20, 20, 20),
        )

    contact_sheet.save(run_dir / "contact-sheet.png")
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate damaged-image samples and a contact sheet for inspection."
    )
    parser.add_argument("input", type=Path, help="Source image to damage")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("image_destroyer_previews"),
        help="Root directory for preview runs (default: image_destroyer_previews)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of variants to generate (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the first variant; subsequent variants increment it",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=10,
        help="Number of contact-sheet columns (default: 10)",
    )
    arguments = parser.parse_args()

    if arguments.count < 1:
        parser.error("--count must be at least 1")
    if arguments.columns < 1:
        parser.error("--columns must be at least 1")

    run_dir = create_previews(
        arguments.input,
        arguments.output_dir,
        count=arguments.count,
        seed=arguments.seed,
        columns=arguments.columns,
    )

    print(f"Generated {arguments.count} variants in {run_dir}")
    print(f"Contact sheet: {run_dir / 'contact-sheet.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
