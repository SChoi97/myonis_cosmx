"""Mask-aware GLCMs with fixed acquisition-wide quantization."""
from __future__ import annotations

import numpy as np


def texture(image, mask, settings, limits):
    yy, xx = np.nonzero(mask)
    result = {"entropy": np.nan}
    distances = settings["distances_px"]
    for d in distances:
        result.update({f"glcm_{p}_d{d}px": np.nan for p in ["contrast", "homogeneity", "energy", "correlation"]})
        result[f"glcm_valid_pairs_d{d}px"] = 0
    if len(yy) < settings["min_pixels"]:
        return result
    image = image[yy.min():yy.max()+1, xx.min():xx.max()+1]
    mask = mask[yy.min():yy.max()+1, xx.min():xx.max()+1]
    mask = mask & np.isfinite(image)
    if mask.sum() < settings["min_pixels"]:
        return result
    levels = int(settings["levels"])
    low, high = limits
    q = np.floor(np.clip((np.nan_to_num(image, nan=low)-low)/(high-low), 0, 1)*(levels-1)).astype(np.int32)
    hist = np.bincount(q[mask], minlength=levels).astype(float)
    probabilities = hist[hist > 0] / hist.sum()
    result["entropy"] = float(-(probabilities*np.log2(probabilities)).sum())
    h, w = mask.shape
    indices = np.arange(levels)
    diff2 = (indices[:, None] - indices[None, :])**2
    for d in distances:
        matrix = np.zeros((levels, levels), dtype=float)
        pairs = 0
        for dy, dx in [(0, d), (d, 0), (d, d), (-d, d)]:
            ya, yb = max(0, -dy), min(h, h-dy)
            xa, xb = max(0, -dx), min(w, w-dx)
            if ya >= yb or xa >= xb:
                continue
            a, b = (slice(ya, yb), slice(xa, xb)), (slice(ya+dy, yb+dy), slice(xa+dx, xb+dx))
            valid = mask[a] & mask[b]
            pairs += int(valid.sum())
            matrix += np.bincount(q[a][valid]*levels+q[b][valid], minlength=levels**2).reshape(levels, levels)
        result[f"glcm_valid_pairs_d{d}px"] = pairs
        if pairs < settings["min_pairs"]:
            continue
        matrix += matrix.T.copy()
        matrix /= matrix.sum()
        pi, pj = matrix.sum(axis=1), matrix.sum(axis=0)
        mi, mj = float(pi @ indices), float(pj @ indices)
        si, sj = np.sqrt(pi @ ((indices-mi)**2)), np.sqrt(pj @ ((indices-mj)**2))
        correlation = float((matrix*(indices[:, None]-mi)*(indices[None, :]-mj)).sum()/(si*sj)) if si*sj > 0 else np.nan
        result.update({f"glcm_contrast_d{d}px": float((matrix*diff2).sum()),
                       f"glcm_homogeneity_d{d}px": float((matrix/(1+diff2)).sum()),
                       f"glcm_energy_d{d}px": float(np.sqrt((matrix**2).sum())),
                       f"glcm_correlation_d{d}px": correlation})
    return result
