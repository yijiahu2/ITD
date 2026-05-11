from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio


def write_semantic_prior_outputs(pred: dict[str, Any], output_dir: str | Path, save_prob_tif: bool) -> dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mod = pred["module"]
    mask = pred["mask"]
    prob = pred["probability"]

    out_tif = out_dir / "M_sem.tif"
    out_png = out_dir / "M_sem.png"
    out_shp = out_dir / "M_sem.shp"
    mod.write_tif(str(out_tif), mask, pred["profile"])
    mod.mask_to_shp(
        mask=mask,
        transform=pred["transform"],
        crs=pred["crs"],
        out_shp=str(out_shp),
        min_area_m2=mod.MIN_AREA_M2,
        simplify_tol=mod.SIMPLIFY_TOL,
    )
    mod.cv2.imwrite(str(out_png), (mask * 255).astype(mod.np.uint8))

    outputs = {
        "m_sem_tif": str(out_tif),
        "m_sem_png": str(out_png),
        "m_sem_shp": str(out_shp),
    }

    if save_prob_tif:
        prob_tif = out_dir / "M_sem_prob.tif"
        profile = pred["tiff_profile_uint8"].copy()
        profile.update(dtype=rasterio.float32, nodata=0.0)
        with rasterio.open(prob_tif, "w", **profile) as dst:
            dst.write(prob.astype(np.float32), 1)
        outputs["m_sem_prob_tif"] = str(prob_tif)

    return outputs
