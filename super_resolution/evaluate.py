"""Evaluation script: metrics, fixed visual grid, and throughput (plan section 11).

Usage:
    python -m super_resolution.evaluate \
        --stage-config experiments_data/<stage_id>/<name>_config.pkl \
        [--output-dir eval_output] [--max-images 16]
"""

import argparse
import math
import time
from pathlib import Path

import torch
from einops import rearrange
from torchvision.utils import save_image

from super_resolution.configs.elements.stage import Stage
from super_resolution.metrics.fidelity import psnr, ssim, y_psnr, y_ssim
from super_resolution.metrics.frequency import band_errors, block_boundary_error


@torch.no_grad()
def evaluate_generator(
    generator: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    scale: int = 4,
    max_images: int | None = None,
) -> tuple[dict[str, float], list[torch.Tensor]]:
    """RGB/Y PSNR+SSIM with 4-pixel shave, plus DCT diagnostics. Images must be
    preprocessed to [0, 1] float by the caller's pipeline before lr/hr split."""
    generator.eval()
    totals: dict[str, float] = {}
    n_images = 0
    grid_images: list[torch.Tensor] = []

    for batch in dataloader:
        hr = batch.to(device).float() / 255.0 if batch.dtype == torch.uint8 else batch.to(device).float()
        if hr.ndim == 5:
            hr = rearrange(hr, "b n c h w -> (b n) c h w")
        if max_images is not None and n_images + hr.shape[0] > max_images:
            keep = max_images - n_images
            hr = hr[:keep]
        if hr.shape[0] == 0:
            break
        lr = torch.nn.functional.interpolate(
            hr, scale_factor=1 / scale, mode="bicubic", align_corners=False
        ).clamp(0, 1)
        sr = generator(lr)

        totals["psnr_rgb"] = totals.get("psnr_rgb", 0.0) + psnr(sr, hr).item() * hr.shape[0]
        totals["ssim_rgb"] = totals.get("ssim_rgb", 0.0) + ssim(sr, hr).item() * hr.shape[0]
        totals["psnr_y"] = totals.get("psnr_y", 0.0) + y_psnr(sr, hr).item() * hr.shape[0]
        totals["ssim_y"] = totals.get("ssim_y", 0.0) + y_ssim(sr, hr).item() * hr.shape[0]
        for key, value in band_errors(sr, hr).items():
            totals[f"dct_{key}"] = totals.get(f"dct_{key}", 0.0) + value * hr.shape[0]
        for key, value in block_boundary_error(sr, hr).items():
            totals[f"{key}"] = totals.get(key, 0.0) + value * hr.shape[0]
        if len(grid_images) < 8:
            grid_images.append(
                torch.cat([lr[:1], sr[:1].clamp(0, 1), hr[:1]]).cpu()
            )
        n_images += hr.shape[0]
        if max_images is not None and n_images >= max_images:
            break

    metrics = {key: value / n_images for key, value in totals.items()}
    metrics["n_images"] = float(n_images)
    return metrics, grid_images


@torch.no_grad()
def measure_throughput(
    generator: torch.nn.Module,
    device: str,
    batch_size: int = 16,
    lr_size: int = 24,
    n_warmup: int = 10,
    n_steps: int = 50,
) -> dict[str, float]:
    """Measured inference images/s at a fixed input size, CUDA-synchronized."""
    generator.eval()
    x = torch.rand(batch_size, 3, lr_size, lr_size, device=device)
    for _ in range(n_warmup):
        generator(x)
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_steps):
        generator(x)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return {
        "images_per_s": batch_size * n_steps / elapsed,
        "step_time_ms": elapsed / n_steps * 1000,
        "trainable_parameters": sum(p.numel() for p in generator.parameters() if p.requires_grad),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("eval_output"))
    parser.add_argument("--max-images", type=int, default=16)
    parser.add_argument("--scale", type=int, default=2)
    args = parser.parse_args()

    stage: Stage = Stage.from_stage_config(args.stage_config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = stage.generator_model.model_config.build().to(device)
    state = stage.load_state()
    generator.load_state_dict(state["generator"])

    _, _, eval_dataset = stage.dataset.load_datasets()
    loader = torch.utils.data.DataLoader(eval_dataset, batch_size=4)
    metrics, grid_images = evaluate_generator(
        generator, loader, device, scale=args.scale, max_images=args.max_images
    )
    metrics.update(measure_throughput(generator, device))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in sorted(metrics.items()):
        print(f"{name}: {value:.4f}")
    if grid_images:
        rows = [torch.cat([img for img in images], dim=2) for images in grid_images]
        save_image(torch.cat(rows, dim=1), args.output_dir / "visual_grid.png", nrow=1)


if __name__ == "__main__":
    main()
