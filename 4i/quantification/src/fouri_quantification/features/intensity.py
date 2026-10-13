from __future__ import annotations

import numpy as np

INTENSITY_STATS = ["mean", "median", "integrated", "std", "p10", "p90", "p95", "min", "max"]


def intensity(image, mask):
    values = np.asarray(image[mask], dtype=np.float64)
    count = values.size
    finite = values[np.isfinite(values)]
    result = dict(n_pixels=int(count), n_valid_pixels=int(finite.size), valid_fraction=float(finite.size/count) if count else np.nan)
    # Incomplete pixel coverage is flagged; never silently convert missing pixels to zero.
    if count == 0 or finite.size != count:
        result.update({key: np.nan for key in INTENSITY_STATS})
        return result
    q10, q90, q95 = np.percentile(values, [10, 90, 95])
    result.update(mean=float(values.mean()), median=float(np.median(values)), integrated=float(values.sum()),
                  std=float(values.std()), p10=float(q10), p90=float(q90), p95=float(q95),
                  min=float(values.min()), max=float(values.max()))
    return result
