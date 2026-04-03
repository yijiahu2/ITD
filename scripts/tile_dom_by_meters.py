from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import rasterio
from rasterio.windows import Window

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ITD_agent.planning.agent.config_builder import load_yaml


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"value must be > 0, got {value}")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"value must be >= 0, got {value}")
    return parsed


def _window_bounds(src: rasterio.io.DatasetReader, window: Window) -> tuple[float, float, float, float]:
    bounds = rasterio.windows.bounds(window, src.transform)
    return float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3])


def _ensure_north_up(src: rasterio.io.DatasetReader) -> None:
    transform = src.transform
    if not math.isclose(transform.b, 0.0, abs_tol=1e-9) or not math.isclose(transform.d, 0.0, abs_tol=1e-9):
        raise ValueError("rotated rasters are not supported by this tiler")


def _to_pixels(size_m: float, resolution: float) -> int:
    return max(1, int(round(size_m / resolution)))


def _load_config_defaults(config_path: str | None) -> dict:
    if not config_path:
        return {}
    cfg = load_yaml(config_path) or {}
    defaults = {}
    tiling_cfg = cfg.get("tiling")
    if isinstance(tiling_cfg, dict):
        defaults.update(tiling_cfg)
    if "input_image" in cfg and "input" not in defaults:
        defaults["input"] = cfg["input_image"]
    return defaults


def tile_raster(
    input_path: str,
    out_dir: str,
    tile_size_m: float,
    overlap_m: float,
    prefix: str | None = None,
    skip_empty: bool = True,
    max_tiles: int | None = None,
) -> dict:
    input_path = str(Path(input_path).resolve())
    out_root = Path(out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    tiles_dir = out_root / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(input_path) as src:
        _ensure_north_up(src)

        res_x = abs(float(src.transform.a))
        res_y = abs(float(src.transform.e))
        tile_width_px = _to_pixels(tile_size_m, res_x)
        tile_height_px = _to_pixels(tile_size_m, res_y)
        overlap_x_px = int(round(overlap_m / res_x))
        overlap_y_px = int(round(overlap_m / res_y))
        step_x = tile_width_px - overlap_x_px
        step_y = tile_height_px - overlap_y_px

        if step_x <= 0 or step_y <= 0:
            raise ValueError(
                "overlap is too large for tile size: "
                f"tile=({tile_width_px},{tile_height_px}) px, overlap=({overlap_x_px},{overlap_y_px}) px"
            )

        tile_prefix = prefix or Path(input_path).stem
        manifest_path = out_root / "tile_index.csv"
        summary_path = out_root / "tile_summary.json"

        profile = src.profile.copy()
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
        profile.update(compress="LZW", tiled=False)

        if src.driver == "GTiff":
            profile["BIGTIFF"] = "IF_SAFER"

        fieldnames = [
            "tile_id",
            "row",
            "col",
            "path",
            "x_off",
            "y_off",
            "width_px",
            "height_px",
            "left",
            "bottom",
            "right",
            "top",
        ]

        generated_tiles = 0
        skipped_tiles = 0
        tile_records: list[dict[str, object]] = []

        with open(manifest_path, "w", encoding="utf-8", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            row_id = 0
            stop = False
            for y_off in range(0, src.height, step_y):
                col_id = 0
                for x_off in range(0, src.width, step_x):
                    width = min(tile_width_px, src.width - x_off)
                    height = min(tile_height_px, src.height - y_off)
                    window = Window(col_off=x_off, row_off=y_off, width=width, height=height)

                    if skip_empty:
                        mask = src.dataset_mask(window=window)
                        if mask.size == 0 or int(mask.max()) == 0:
                            skipped_tiles += 1
                            col_id += 1
                            continue

                    tile_id = f"r{row_id:04d}_c{col_id:04d}"
                    tile_path = tiles_dir / f"{tile_prefix}_{tile_id}.tif"
                    data = src.read(window=window)
                    tile_profile = profile.copy()
                    tile_profile.update(
                        width=width,
                        height=height,
                        transform=rasterio.windows.transform(window, src.transform),
                    )

                    with rasterio.open(tile_path, "w", **tile_profile) as dst:
                        dst.write(data)

                    left, bottom, right, top = _window_bounds(src, window)
                    record = {
                        "tile_id": tile_id,
                        "row": row_id,
                        "col": col_id,
                        "path": str(tile_path),
                        "x_off": x_off,
                        "y_off": y_off,
                        "width_px": width,
                        "height_px": height,
                        "left": left,
                        "bottom": bottom,
                        "right": right,
                        "top": top,
                    }
                    writer.writerow(record)
                    tile_records.append(record)
                    generated_tiles += 1

                    if max_tiles is not None and generated_tiles >= max_tiles:
                        stop = True
                        break

                    col_id += 1

                row_id += 1
                if stop:
                    break

        summary = {
            "input_path": input_path,
            "out_dir": str(out_root),
            "tiles_dir": str(tiles_dir),
            "manifest_csv": str(manifest_path),
            "tile_size_m": tile_size_m,
            "overlap_m": overlap_m,
            "resolution_x": res_x,
            "resolution_y": res_y,
            "tile_width_px": tile_width_px,
            "tile_height_px": tile_height_px,
            "overlap_x_px": overlap_x_px,
            "overlap_y_px": overlap_y_px,
            "step_x_px": step_x,
            "step_y_px": step_y,
            "generated_tiles": generated_tiles,
            "skipped_tiles": skipped_tiles,
            "max_tiles": max_tiles,
            "tiles": tile_records,
        }

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        return summary


def main() -> None:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None, help="Optional YAML config file.")
    config_args, remaining = config_parser.parse_known_args()
    defaults = _load_config_defaults(config_args.config)

    parser = argparse.ArgumentParser(
        description="Tile a large DOM by physical size in meters.",
        parents=[config_parser],
    )
    parser.add_argument("--input", default=None, help="Input GeoTIFF path.")
    parser.add_argument("--out_dir", default=None, help="Output root directory.")
    parser.add_argument("--tile_size_m", type=_positive_float, default=30.0, help="Tile size in meters.")
    parser.add_argument(
        "--overlap_m",
        type=_nonnegative_float,
        default=5.0,
        help="Tile overlap in meters. Default keeps merge margins for segmentation.",
    )
    parser.add_argument("--prefix", default=None, help="Optional tile filename prefix.")
    parser.add_argument(
        "--keep_empty_tiles",
        action="store_true",
        default=False,
        help="Write tiles even when the dataset mask is fully empty.",
    )
    parser.add_argument(
        "--max_tiles",
        type=int,
        default=None,
        help="Optional cap for smoke tests. When set, stop after writing this many tiles.",
    )
    parser.set_defaults(**defaults)
    args = parser.parse_args(remaining)
    if not args.input:
        parser.error("--input is required (or provide it in config under input_image / tiling.input)")
    if not args.out_dir:
        parser.error("--out_dir is required (or provide it in config under tiling.out_dir)")

    summary = tile_raster(
        input_path=args.input,
        out_dir=args.out_dir,
        tile_size_m=args.tile_size_m,
        overlap_m=args.overlap_m,
        prefix=args.prefix,
        skip_empty=not args.keep_empty_tiles,
        max_tiles=args.max_tiles,
    )

    print(f"[OK] tiles written: {summary['generated_tiles']}")
    print(f"[OK] skipped empty tiles: {summary['skipped_tiles']}")
    print(f"[OK] manifest: {summary['manifest_csv']}")


if __name__ == "__main__":
    main()
