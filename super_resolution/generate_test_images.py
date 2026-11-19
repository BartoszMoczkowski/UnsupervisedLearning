#!/usr/bin/env python3
"""Generate five-image visual comparisons for every valid saved Stage checkpoint.

Images are sampled once from 1024/eval_1 and reused for every config. Outputs use
    <output-dir>/<experiment>/<stage>/image_1.png ... image_5.png

Usage:
    python -m super_resolution.generate_test_images
    python -m super_resolution.generate_test_images --images /data/wikiart-parquet-images/1024/eval_1 --output-dir test_image_results
"""

import argparse
import random
from pathlib import Path

import torch
from torchvision.transforms import v2 as tv2

from super_resolution.configs.elements.stage import Stage
from super_resolution.generate_test_image import _load_image, generate_test_strip

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
N_IMAGES = 5


def _choose_images(images_root: Path, seed: int) -> list[Path]:
    files = sorted(
        path
        for path in images_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if len(files) < N_IMAGES:
        raise FileNotFoundError(
            f"Need {N_IMAGES} images under {images_root}, found {len(files)}"
        )
    return random.Random(seed).sample(files, N_IMAGES)


def _load_valid_stage(
    config_path: Path, device: str
) -> tuple[Stage, torch.nn.Module]:
    """Load a config and validate its adjacent state against the configured model."""
    stage: Stage = Stage.from_stage_config(config_path)
    checkpoint_path = config_path.with_name(
        f"{stage.human_readable_name}_state.pt"
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"missing {checkpoint_path.name}")

    # Use the adjacent checkpoint rather than recomputing a stage ID, which can
    # change when config classes gain new fields.
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    generator = stage.generator_model.model_config.build().to(device)
    generator.load_state_dict(state["generator"])
    generator.eval()
    return stage, generator


def _stage_output_dir(
    output_root: Path,
    stage: Stage,
    config_path: Path,
    used_paths: set[Path],
) -> Path:
    stage_name = stage.human_readable_name
    experiment_name = stage_name.split("-", 1)[0]
    output_dir = output_root / experiment_name / stage_name
    if output_dir in used_paths:
        checkpoint_suffix = config_path.parent.name.rsplit("_", 1)[-1]
        output_dir = output_dir.with_name(f"{stage_name}_{checkpoint_suffix}")
    counter = 2
    unique_dir = output_dir
    while unique_dir in used_paths:
        unique_dir = output_dir.with_name(f"{output_dir.name}_{counter}")
        counter += 1
    used_paths.add(unique_dir)
    return unique_dir


def run_batch(
    configs_root: Path,
    images_root: Path,
    output_dir: Path,
    max_side: int = 512,
    seed: int = 43,
    device: str | None = None,
) -> list[Path]:
    config_paths = sorted(configs_root.glob("*/*_config.pkl"))
    if not config_paths:
        raise FileNotFoundError(f"No stage configs under {configs_root}")

    image_paths = _choose_images(images_root, seed)
    images = [_load_image(path) for path in image_paths]
    print("Selected images:")
    for path in image_paths:
        print(f"  {path}")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_paths: list[Path] = []
    used_stage_paths: set[Path] = set()

    for config_path in config_paths:
        print(f"Loading {config_path}")
        try:
            stage, generator = _load_valid_stage(config_path, device)
        except Exception as error:
            print(f"  skipping invalid config/checkpoint: {error}")
            continue

        stage_output = _stage_output_dir(
            output_dir, stage, config_path, used_stage_paths
        )
        stage_output.mkdir(parents=True, exist_ok=True)
        for image_index, img in enumerate(images, start=1):
            strip, _, _ = generate_test_strip(
                stage, generator, img, max_side, device
            )
            output_path = stage_output / f"image_{image_index}.png"
            tv2.functional.to_pil_image(strip).save(output_path)
            output_paths.append(output_path)
        print(f"  saved {N_IMAGES} comparisons -> {stage_output}")

        del generator
        if device == "cuda":
            torch.cuda.empty_cache()

    if not output_paths:
        raise RuntimeError("No valid config/checkpoint pairs produced output")
    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs-root", type=Path, default=Path("experiments_data"))
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("/data/wikiart-parquet-images/1024/eval_1"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("test_image_results")
    )
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()

    outputs = run_batch(
        configs_root=args.configs_root,
        images_root=args.images,
        output_dir=args.output_dir,
        max_side=args.max_side,
        seed=args.seed,
    )
    print(f"Done: {len(outputs)} images written to {args.output_dir}")


if __name__ == "__main__":
    main()
