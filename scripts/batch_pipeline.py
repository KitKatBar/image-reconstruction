"""Batch degradation, interpolation, evaluation, and plotting pipeline.

The script is designed for personal image folders and datasets such as DIV2K.
For each source image and scale factor it:

* saves a pixelated, same-size input image for future neural-network training;
* saves the original high-resolution target image once per split;
* saves the smaller low-resolution image used by interpolation;
* creates bicubic and Lanczos restorations;
* calculates MAE, PSNR, and SSIM;
* writes per-image and aggregate CSV files;
* generates metric plots.

Example:
    python batch_pipeline.py \
        --input-dir data/raw/personal \
        --split-name personal \
        --scales 2 4 8 16

DIV2K training example:
    python batch_pipeline.py \
        --input-dir data/DIV2K_train_HR \
        --split-name train \
        --scales 2 4 8 16
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--split-name",
        default="custom",
        help="Name used in output paths and CSV files, e.g. train, valid, or test",
    )
    parser.add_argument(
        "--scales",
        type=int,
        nargs="+",
        default=[2, 4, 8, 16],
        help="Downscaling factors to process",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for input-target pairs",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/batch_pipeline"),
        help="Directory for metrics, plots, and restored images",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional limit for a quick test run",
    )
    parser.add_argument(
        "--save-restorations",
        action="store_true",
        help="Save bicubic and Lanczos output images for visual inspection",
    )
    return parser.parse_args()


def find_images(input_dir: Path) -> List[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    images = sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"No supported images found under: {input_dir}")
    return images


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def create_degraded_image(
    image: Image.Image, scale: int
) -> tuple[Image.Image, Image.Image]:
    """Return low-resolution and same-size pixelated images.

    BOX downsampling averages pixels into blocks. Nearest-neighbor enlargement
    makes the information loss visible while preserving the original aspect
    ratio. The same-size pixelated image is the future neural-network input.
    """
    if scale < 2:
        raise ValueError(f"Scale must be >= 2, received {scale}")

    width, height = image.size
    low_size = (max(1, width // scale), max(1, height // scale))
    low_resolution = image.resize(low_size, Image.Resampling.BOX)
    pixelated = low_resolution.resize((width, height), Image.Resampling.NEAREST)
    return low_resolution, pixelated


def restore_with_interpolation(
    low_resolution: Image.Image, output_size: tuple[int, int]
) -> Dict[str, Image.Image]:
    return {
        "bicubic": low_resolution.resize(output_size, Image.Resampling.BICUBIC),
        "lanczos": low_resolution.resize(output_size, Image.Resampling.LANCZOS),
    }


def to_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image, dtype=np.float32) / 255.0


def calculate_metrics(
    target: Image.Image, prediction: Image.Image
) -> Dict[str, float]:
    target_array = to_array(target)
    prediction_array = to_array(prediction)
    difference = target_array - prediction_array
    mae = float(np.mean(np.abs(difference)))
    mse = float(np.mean(np.square(difference)))
    psnr = float("inf") if mse == 0 else float(10.0 * np.log10(1.0 / mse))
    ssim = float(
        structural_similarity(
            target_array,
            prediction_array,
            channel_axis=2,
            data_range=1.0,
        )
    )
    return {"mae": mae, "psnr_db": psnr, "ssim": ssim}


def safe_image_id(image_path: Path, input_dir: Path) -> str:
    """Create a collision-resistant filename from a relative image path."""
    relative = image_path.relative_to(input_dir).with_suffix("")
    value = "__".join(relative.parts)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def save_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: Sequence[dict]) -> List[dict]:
    grouped: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["split"], int(row["scale"]), row["method"])].append(row)

    summary = []
    for (split, scale, method), group in sorted(grouped.items()):
        values = {
            metric: [float(row[metric]) for row in group]
            for metric in ("mae", "psnr_db", "ssim")
        }
        result = {
            "split": split,
            "scale": scale,
            "method": method,
            "image_count": len(group),
        }
        for metric, metric_values in values.items():
            result[f"{metric}_mean"] = mean(metric_values)
            result[f"{metric}_std"] = pstdev(metric_values) if len(metric_values) > 1 else 0.0
        summary.append(result)
    return summary


def plot_metric(summary_rows: Sequence[dict], metric: str, output_path: Path) -> None:
    methods = sorted({row["method"] for row in summary_rows})
    figure, axis = plt.subplots(figsize=(8, 5))

    for method in methods:
        rows = sorted(
            (row for row in summary_rows if row["method"] == method),
            key=lambda row: int(row["scale"]),
        )
        scales = [int(row["scale"]) for row in rows]
        averages = [float(row[f"{metric}_mean"]) for row in rows]
        deviations = [float(row[f"{metric}_std"]) for row in rows]
        axis.errorbar(scales, averages, yerr=deviations, marker="o", capsize=4, label=method)

    axis.set_xscale("log", base=2)
    axis.set_xticks(sorted({int(row["scale"]) for row in summary_rows}))
    axis.set_xlabel("Degradation scale factor")
    axis.set_ylabel(metric.upper() if metric != "psnr_db" else "PSNR (dB)")
    axis.set_title(f"{metric.upper()} versus degradation severity")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def write_summary_json(path: Path, summary_rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(list(summary_rows), file, indent=2)


def main() -> None:
    args = parse_args()
    scales = sorted(set(args.scales))
    if any(scale < 2 for scale in scales):
        raise ValueError("Every scale must be at least 2")

    image_paths = find_images(args.input_dir)
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]

    target_dir = args.processed_dir / args.split_name / "targets"
    rows: List[dict] = []

    print(f"Found {len(image_paths)} image(s) under {args.input_dir}")
    print(f"Processing scales: {scales}")

    for index, image_path in enumerate(image_paths, start=1):
        original = load_image(image_path)
        image_id = safe_image_id(image_path, args.input_dir)
        target_path = target_dir / f"{image_id}.png"
        save_image(original, target_path)

        for scale in scales:
            low_resolution, pixelated = create_degraded_image(original, scale)
            restorations = restore_with_interpolation(low_resolution, original.size)
            scale_dir = args.processed_dir / args.split_name / f"scale_x{scale}"

            save_image(pixelated, scale_dir / "inputs" / f"{image_id}.png")
            save_image(low_resolution, scale_dir / "low_resolution" / f"{image_id}.png")

            if args.save_restorations:
                for method, restoration in restorations.items():
                    save_image(
                        restoration,
                        args.results_dir
                        / "restorations"
                        / args.split_name
                        / f"scale_x{scale}"
                        / method
                        / f"{image_id}.png",
                    )

            for method, restoration in restorations.items():
                metric_values = calculate_metrics(original, restoration)
                rows.append(
                    {
                        "split": args.split_name,
                        "image_id": image_id,
                        "source_path": str(image_path),
                        "width": original.size[0],
                        "height": original.size[1],
                        "scale": scale,
                        "method": method,
                        **metric_values,
                    }
                )

        print(f"[{index}/{len(image_paths)}] {image_path.name}")

    detail_fields = [
        "split",
        "image_id",
        "source_path",
        "width",
        "height",
        "scale",
        "method",
        "mae",
        "psnr_db",
        "ssim",
    ]
    summary_rows = summarize(rows)
    summary_fields = [
        "split",
        "scale",
        "method",
        "image_count",
        "mae_mean",
        "mae_std",
        "psnr_db_mean",
        "psnr_db_std",
        "ssim_mean",
        "ssim_std",
    ]

    write_csv(args.results_dir / "metrics_per_image.csv", rows, detail_fields)
    write_csv(args.results_dir / "summary_metrics.csv", summary_rows, summary_fields)
    write_summary_json(args.results_dir / "summary_metrics.json", summary_rows)

    for metric in ("mae", "psnr_db", "ssim"):
        plot_metric(summary_rows, metric, args.results_dir / "plots" / f"{metric}_by_scale.png")

    print(f"\nProcessed {len(image_paths)} image(s).")
    print(f"Pair files saved under: {args.processed_dir / args.split_name}")
    print(f"Metrics and plots saved under: {args.results_dir}")


if __name__ == "__main__":
    main()
