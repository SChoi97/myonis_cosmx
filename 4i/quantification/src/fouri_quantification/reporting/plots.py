from pathlib import Path

import numpy as np


def save_overlay(image, nuclei, myotubes, nucleus_rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axis = plt.subplots(figsize=(8, 8), dpi=150)
    finite = image[np.isfinite(image)]
    lo, hi = np.percentile(finite, [1, 99.5]) if finite.size else (0, 1)
    axis.imshow(image, cmap="gray", vmin=lo, vmax=max(hi, lo+1))
    for obj in myotubes:
        polygon = obj["local_polygon"]
        polygon = np.vstack([polygon, polygon[0]])
        axis.plot(polygon[:, 0], polygon[:, 1], color="#f6ad55", lw=1)
        axis.text(obj["centroid_local_x"], obj["centroid_local_y"], f"M{obj['local_id']}", color="#f6ad55", fontsize=8)
    for obj, row in zip(nuclei, nucleus_rows):
        polygon = obj["local_polygon"]
        polygon = np.vstack([polygon, polygon[0]])
        color = "#68d391" if row["parent_myotube_uid"] else "#fc8181"
        axis.plot(polygon[:, 0], polygon[:, 1], color=color, lw=.7)
        axis.text(obj["centroid_local_x"], obj["centroid_local_y"], f"N{obj['local_id']}", color=color, fontsize=5)
    axis.set_title("MYONIS 4i | green: assigned nucleus | red: unassigned | orange: myotube", fontsize=9)
    axis.axis("off")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
