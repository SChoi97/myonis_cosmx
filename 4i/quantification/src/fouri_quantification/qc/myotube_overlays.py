"""Display source myotubes over every registered marker channel, without quantification."""
from __future__ import annotations

import os
from pathlib import Path
import re

ACQUISITION_PREFIXES = ("measurement-", "source-", "fluor-", "marker-")


def discover_channels(run_dir, rounds, reference_image=None):
    """Resolve an example patch directly before falling back to a directory listing."""
    channels = []
    for folder in sorted((Path(run_dir) / "png_8bit").iterdir()):
        match = re.fullmatch(r"measurement_(\d+)_(.+)", folder.name)
        if not match or int(match[1]) not in rounds or match[2] == "dapi" or not folder.is_dir():
            continue
        template = None
        parts = match[2].split("_", 1)
        if reference_image is not None and len(parts) == 2:
            # Phenix exports use four source channels. Verify the actual file;
            # never assume a fluorophore implies a particular source channel.
            for ch in range(1, 5):
                replacements = dict(zip(ACQUISITION_PREFIXES,
                                        (f"measurement-{int(match[1]):02d}", f"source-ch{ch:02d}",
                                         f"fluor-{parts[0]}", f"marker-{parts[1]}")))
                tokens = [next((replacements[p] for p in ACQUISITION_PREFIXES if token.startswith(p)), token)
                          for token in Path(reference_image).stem.split("__")]
                candidate = "__".join(tokens)+".png"
                if (folder / candidate).is_file():
                    template = candidate
                    break
        if template is None:
            with os.scandir(folder) as entries:
                template = next((entry.name for entry in entries if entry.name.endswith(".png") and not entry.name.startswith(".")), None)
        if template is None:
            continue
        tokens = {prefix: next((token for token in Path(template).stem.split("__") if token.startswith(prefix)), None)
                  for prefix in ACQUISITION_PREFIXES}
        if any(value is None for value in tokens.values()):
            raise ValueError(f"Cannot resolve acquisition tokens from {folder / template}")
        channels.append(dict(acquisition=folder.name, round=int(match[1]), directory=str(folder),
                             tokens=tokens, marker=tokens["marker-"].removeprefix("marker-"),
                             fluor=tokens["fluor-"].removeprefix("fluor-")))
    return channels


def channel_path(dapi_path, channel):
    tokens = []
    for token in Path(dapi_path).stem.split("__"):
        prefix = next((prefix for prefix in ACQUISITION_PREFIXES if token.startswith(prefix)), None)
        tokens.append(channel["tokens"][prefix] if prefix is not None else token)
    return Path(channel["directory"]) / ("__".join(tokens) + ".png")


def save_myotube_overlay(result, path, args, mvt, plt, np, apply_translation):
    """Shift source geometry once; shift every channel with its own round's DAPI transform."""
    from myonis_vis.utils.images import prepare_image
    field, ref = result["field"], result["reference_round"]
    channels = args.myotube_channels
    source = next(channel for channel in channels if channel["acquisition"] == args.myotube_acquisition)
    source_round = source["round"]
    source_image = channel_path(field["paths"][source_round], source)
    candidates = [Path(args.myotube_contour_dir) / (source_image.stem + ".txt"),
                  Path(args.myotube_contour_dir) / (Path(field["paths"][source_round]).stem + ".txt")]
    present = [candidate for candidate in candidates if candidate.is_file()]
    if len(present) != 1:
        raise ValueError(f"Expected one MyHC/DAPI-named myotube contour file for {source_image.name}; found {len(present)}")
    contour_path = present[0]
    contours = mvt.read_yolo_contours(contour_path, field["shape"])
    identity = dict(shift_y=0.0, shift_x=0.0)
    geometry_shift = result["registrations"].get(source_round, identity)
    filter_rounds = set(result["comparison_rounds"]) | {ref}
    columns = 4
    rows = (len(channels)+2+columns-1)//columns
    fig, axes = plt.subplots(rows, columns, figsize=(16, 4*rows), squeeze=False, layout="constrained")
    temporary = path.with_suffix(".tmp.png")
    missing = []

    def show(axis, image, title, shift_y, shift_x):
        axis.imshow(prepare_image(image, intensity_percentiles=tuple(args.intensity_percentiles)))
        for contour in contours:
            for ring in getattr(contour, "rings", (contour,)):
                points = np.asarray(ring).reshape(-1, 2)
                if len(points):
                    points = np.vstack((points, points[0]))
                    axis.plot(points[:, 0]+shift_x, points[:, 1]+shift_y, color="#00ffff", linewidth=0.65, alpha=0.9)
        axis.set_title(title, fontsize=9)

    try:
        for axis in axes.flat:
            axis.set_axis_off()
        original = mvt.read_image(source_image)
        if original.shape != field["shape"]:
            raise ValueError(f"MyHC image grid differs from reference DAPI: {source_image}")
        show(axes.flat[0], original, f"MyHC source: round {source_round}\nOriginal segmentation ({len(contours)} myotubes)", 0, 0)
        show(axes.flat[1], field["images"][ref], f"Reference DAPI: round {ref}\nMyotubes in the reference frame",
             geometry_shift["shift_y"], geometry_shift["shift_x"])
        for axis, channel in zip(list(axes.flat)[2:], channels):
            r = channel["round"]
            registration = result["registrations"].get(r, identity)
            image_path = channel_path(field["paths"][r], channel)
            title = f"Round {r} · {channel['marker']} · {channel['fluor']}"
            if r not in filter_rounds:
                title += "\nEXCLUDED FROM DAPI FILTERING"
            title += f"\ndy={registration['shift_y']:.2f}, dx={registration['shift_x']:.2f} px"
            if not image_path.is_file():
                missing.append(str(image_path))
                axis.set_title(title, fontsize=9)
                axis.text(0.5, 0.5, "Image unavailable", ha="center", va="center", transform=axis.transAxes)
                continue
            image = mvt.read_image(image_path)
            if image.shape != field["shape"]:
                raise ValueError(f"Marker image grid differs from reference DAPI: {image_path}")
            registered = apply_translation(image, registration["shift_y"], registration["shift_x"])
            show(axis, registered, title, geometry_shift["shift_y"], geometry_shift["shift_x"])
        fig.suptitle(f"{field['label']} · Myotube contours (cyan) over registered marker channels\n"
                     f"DAPI reference: round {ref} · marker images retain their own intensities", fontsize=12)
        fig.savefig(temporary, dpi=args.dpi, facecolor="white")
        temporary.replace(path)
    finally:
        plt.close(fig)
        temporary.unlink(missing_ok=True)
    return dict(contour_path=str(contour_path), source_image=str(source_image),
                source_round=source_round, reference_round=ref, n_myotubes=len(contours),
                geometry_shift_y=geometry_shift["shift_y"], geometry_shift_x=geometry_shift["shift_x"],
                excluded_filter_rounds=sorted(set(field["images"])-filter_rounds),
                missing_images=missing, acquisitions=[channel["acquisition"] for channel in channels])
