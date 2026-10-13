"""Selected rounds must participate in matching, filtering, plots and QC exports."""
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fouri_quantification.qc import run_4i_nucleus_segmentation_qc as qc


def mosaic_name(round_number, patch_number=1):
    return (f"plate-sc_rep5-6__rep-05__well-r02c02__measurement-{round_number:02d}"
            f"__source-ch01__fluor-dapi__marker-dapi__patch_{patch_number:04d}"
            f"__pr01-pc{patch_number:02d}__refy-000000-refx-{(patch_number-1)*96:06d}")


class StandaloneDiscoveryTest(unittest.TestCase):
    def test_twelve_round_matching_sampling_and_png_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contours = root / "reference_contours"
            contours.mkdir()
            for r in range(1, 13):
                folder = root / "png_8bit" / f"measurement_{r:02d}_dapi"
                folder.mkdir(parents=True)
                for p in range(1, 5):
                    if r == 12 and p == 4:
                        continue
                    (folder / (mosaic_name(r, p)+".png")).touch()
                    if r == 2:
                        (contours / (mosaic_name(r, p)+".txt")).touch()
            args = qc.parse_args(["--datapath", str(root / "png_8bit"),
                                  "--rounds", *map(str, range(1, 13)),
                                  "--registration_metric", "ssim"])
            self.assertEqual(args.run_dir, root.resolve())
            self.assertEqual(args.rounds, list(range(1, 13)))
            with contextlib.redirect_stdout(io.StringIO()):
                indices, common, discovery = qc.discover_patches(
                    root, reference_only=2, reference_round=2,
                    reference_contour_dir=contours, rounds=args.rounds)
            self.assertEqual(len(common), 3)
            self.assertEqual(discovery["incomplete_patches"][0]["missing_rounds"], [12])
            self.assertIsNone(indices[12][common[0]][1])
            selected, _ = qc.select_patches(common, "all", 42, None, 2)
            again, _ = qc.select_patches(list(reversed(common)), "all", 42, None, 2)
            self.assertEqual(selected, again)
            self.assertEqual(len(selected), 2)

    def test_round_validation_and_original_default(self):
        self.assertEqual(qc.parse_args([]).rounds, list(range(1, 9)))
        self.assertEqual(qc.parse_args(["--rounds", "9", "12", "--reference-round", "12"]).reference_round, 12)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                qc.parse_args(["--rounds", "1", "12"])
            with self.assertRaises(SystemExit):
                qc.parse_args(["--rounds", "2", "2"])
            for rounds in ([2], [2, 13], [2, 2, 3], [1, 3]):
                with self.assertRaises(SystemExit):
                    qc.parse_args(["--registration_metric", "ssim", "--filter-rounds", *map(str, rounds)])


class StandaloneSSIMTest(unittest.TestCase):
    def test_late_round_loss_saved_transforms_and_every_panel(self):
        import numpy as np
        from PIL import Image
        from scipy.ndimage import gaussian_filter, shift

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contours = root / "reference_contours"
            contours.mkdir()
            size = 96
            yy, xx = np.indices((size, size))
            background = gaussian_filter(np.random.default_rng(31).normal(0, 1, (size, size)), 1)*12 + 15
            spots = [180*np.exp(-((xx-x)**2+(yy-y)**2)/50)
                     for y, x in [(30, 30), (60, 62), (43, 82), (76, 17)]]
            fixed = np.clip(background + sum(spots), 1, 255).astype(np.uint8)
            for r in range(1, 13):
                image = fixed if r != 12 else np.clip(background + spots[0] + spots[2] + spots[3], 1, 255).astype(np.uint8)
                if r != 2:
                    image = shift(image, (-1, 2), order=0, mode="constant", cval=0)
                folder = root / "png_8bit" / f"measurement_{r:02d}_dapi"
                folder.mkdir(parents=True)
                Image.fromarray(image).save(folder / (mosaic_name(r)+".png"))
            boxes = [(21, 21, 39, 39), (53, 51, 71, 69), (0, 40, 5, 49)]
            lines = []
            for x0, y0, x1, y1 in boxes:
                values = [x0/size, y0/size, x1/size, y0/size, x1/size, y1/size, x0/size, y1/size]
                lines.append("0 " + " ".join(map(str, values)))
            (contours / (mosaic_name(2)+".txt")).write_text("\n".join(lines)+"\n")
            provenance = root / "provenance"
            provenance.mkdir()
            (provenance / "pipeline_config.json").write_text(json.dumps({"registration": {
                "downsample": 1, "upsample_factor": 1, "min_global_correlation": 0.25,
                "patch_level_min_overlap_fraction": 0.8, "patch_level_max_shift_px": 10}}))
            output = root / "output"
            qc.initialize_worker(qc.default_tools_dir())
            titles = {}
            original_savefig = qc.plt.Figure.savefig

            def capture(figure, path, *args, **kwargs):
                titles[Path(path).parent.name] = [axis.get_title() for axis in figure.axes]
                return original_savefig(figure, path, *args, **kwargs)

            with patch.object(qc.plt.Figure, "savefig", new=capture):
                status = qc.main(["--datapath", str(root), "--rounds", *map(str, range(1, 13)),
                                  "--reference-contour-dir", str(contours),
                                  "--registration_metric", "ssim", "--registration-alignment-panels",
                                  "--skip-dense-maps", "--n_images", "1", "--dpi", "30",
                                  "--render-scale", "1", "--output-dir", str(output)])
            self.assertEqual(status, 0)
            record = json.loads(next((output / "assignments").glob("*.json")).read_text())
            self.assertEqual(len(record["comparison_rounds"]), 11)
            self.assertEqual(record["statuses"], ["kept", "filtered_change", "not_evaluable"])
            for r in record["comparison_rounds"]:
                registration = record["rounds"][str(r)]["registration"]
                self.assertEqual((registration["shift_y"], registration["shift_x"]), (1, -2))
            for folder in ["plots", "registration_overlays", "registration_alignment"]:
                self.assertTrue(any("Round 12" in title for title in titles[folder]), folder)
            self.assertEqual(len(list(output.rglob("*.png"))), 4)
            self.assertFalse(list(output.rglob("*.npz")))
            with (output / "round_summary.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 12)
            with (output / "myonis_4i/round_qc.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 36)

            # A nucleus lost only in an excluded round must remain eligible.
            image = np.clip(background + spots[1] + spots[2] + spots[3], 1, 255).astype(np.uint8)
            Image.fromarray(shift(image, (-1, 2), order=0, mode="constant", cval=0)).save(
                root / "png_8bit/measurement_07_dapi" / (mosaic_name(7)+".png"))
            channels = [(3, "myhc", "555"), (7, "p_mlkl", "647"),
                        (8, "pten", "488"), (9, "p4ebp1", "555"), (12, "atf6", "647")]
            myotubes = root / "myotubes"
            myotubes.mkdir()
            for r, marker, fluor in channels:
                folder = root / "png_8bit" / f"measurement_{r:02d}_{fluor}_{marker}"
                folder.mkdir()
                name = mosaic_name(r).replace("source-ch01__fluor-dapi__marker-dapi",
                                             f"source-ch03__fluor-{fluor}__marker-{marker}")
                Image.fromarray(shift(fixed, (-1, 2), order=0, mode="constant", cval=0)).save(folder / (name+".png"))
                if marker == "myhc":
                    # Native round-3 geometry; after dy=1, dx=-2 its box is (20,20)-(40,40).
                    coords = [22/size, 19/size, 42/size, 19/size, 42/size, 39/size, 22/size, 39/size]
                    (myotubes / (name+".txt")).write_text("0 " + " ".join(map(str, coords))+"\n")
            output = root / "excluded_output"
            outlines = []

            def capture_myotubes(figure, path, *args, **kwargs):
                if Path(path).parent.name == "myotube_overlays":
                    outlines.extend((line.get_xdata(), line.get_ydata()) for line in figure.axes[1].lines)
                return capture(figure, path, *args, **kwargs)

            with patch.object(qc.plt.Figure, "savefig", new=capture_myotubes):
                status = qc.main(["--datapath", str(root), "--rounds", *map(str, range(1, 13)),
                                  "--filter-rounds", "1", "2", "3", "4", "5", "6", "10", "11", "12",
                                  "--reference-contour-dir", str(contours),
                                  "--registration_metric", "ssim", "--registration-alignment-panels",
                                  "--myotube-contour-dir", str(myotubes),
                                  "--myotube-acquisition", "measurement_03_555_myhc",
                                  "--skip-dense-maps", "--n_images", "1", "--dpi", "30",
                                  "--render-scale", "1", "--output-dir", str(output)])
            self.assertEqual(status, 0)
            record = json.loads(next((output / "assignments").glob("*.json")).read_text())
            self.assertEqual(record["comparison_rounds"], [1, 3, 4, 5, 6, 10, 11, 12])
            self.assertEqual(record["excluded_filter_rounds"], [7, 8, 9])
            self.assertEqual(record["statuses"], ["kept", "filtered_change", "not_evaluable"])
            self.assertEqual(len(record["rounds"]), 12)
            for r in (7, 8, 9):
                self.assertFalse(record["rounds"][str(r)]["used_for_filtering"])
                self.assertEqual(record["rounds"][str(r)]["mean_change_score"], [None]*3)
                self.assertTrue(any(f"Round {r}" in title and "EXCLUDED FROM DAPI FILTERING" in title
                                    for title in titles["myotube_overlays"]))
            overlay = record["myotube_overlay"]
            self.assertEqual((overlay["geometry_shift_y"], overlay["geometry_shift_x"]), (1, -2))
            self.assertEqual(overlay["n_myotubes"], 1)
            self.assertEqual(overlay["missing_images"], [])
            self.assertEqual(len(outlines), 1)
            self.assertAlmostEqual(min(outlines[0][0]), 20)
            self.assertAlmostEqual(min(outlines[0][1]), 20)
            self.assertEqual(len(list(output.rglob("*.png"))), 5)
            self.assertFalse(list(output.rglob("*.npz")))
            with (output / "myonis_4i/nucleus_qc.csv").open() as handle:
                rows = list(csv.DictReader(handle))
                self.assertTrue(all(row["qc_n_comparison_rounds"] == "8" for row in rows))
                self.assertEqual(rows[0]["is_persistent"], "1")
            with (output / "myonis_4i/round_qc.csv").open() as handle:
                rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), 36)
                self.assertTrue(all(row["is_filter_round"] == "0" and row["score"] == ""
                                    for row in rows if int(row["round"]) in (7, 8, 9)))


class StandaloneImportTest(unittest.TestCase):
    def test_import_retains_unscored_rounds_and_rejects_missing_evidence(self):
        import pandas as pd
        from fouri_quantification.inputs.metadata import parse_filename
        from fouri_quantification.inputs.nucleus_qc import load_nucleus_qc

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = ("plate-sc_rep5-6__rep-05__well-r04c02__field-f01__cohort-m01-12__measurement-02"
                    "__source-ch01__fluor-dapi__marker-dapi__m01_12__field_01__patch_0001"
                    "__pr01-pc01__fieldrefy-0000-fieldrefx-0000__refy-000000-refx-000000")
            contour = root / (name+".txt")
            contour.write_text("0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n"
                               "0 0.4 0.4 0.5 0.4 0.5 0.5 0.4 0.5\n")
            source = qc.source_snapshot(root / (name+".png"), contour)
            comparisons = [1, 3, 4, 5, 6, 10, 11, 12]
            record = dict(patch_key="patch", filter_mode="registration_based", reference_round=2,
                          reference_count=2, reference_source=source, shape=[32, 32],
                          comparison_rounds=comparisons, change_threshold=.35, score_metric="ssim",
                          score_aggregation="max", is_edge=[False, False], statuses=["kept", "kept"],
                          reference_area_pixels=[9, 9], kept_reference_ids=[1, 2],
                          rounds={str(r): dict(mean_change_score=[0.1, 0.1] if r in comparisons else
                                              ([0, 0] if r == 2 else [None, None])) for r in range(1, 13)})
            (root / "assignments").mkdir()
            (root / "assignments/patch.json").write_text(json.dumps(record))
            summary = dict(status="complete", settings=dict(reference_round=2, rounds=list(range(1, 13)),
                                                             filter_rounds=[2, *comparisons]),
                           completed=[dict(patch_key="patch", assignments="assignments/patch.json")])
            folder = qc.export_myonis_4i(root, summary)
            layout = dict(plates={"sc_rep5-6": dict(plate_id=1, repeats=[5, 6])},
                          rows={4: dict(cell_line="K32Del", condition="K32Del")})
            meta = parse_filename(name, "fixture", layout)
            config = dict(dataset_id="fixture", plate_layout_data=layout,
                          segmentations=dict(nuclei=dict(acquisition_id="measurement_02_dapi")),
                          nucleus_qc=dict(directory=str(folder), require_all=True))
            images = pd.DataFrame([dict(spatial_key=meta["spatial_key"], acquisition_id="measurement_02_dapi",
                                        image_name=name, patch_size=32)])
            assets = pd.DataFrame([dict(spatial_key=meta["spatial_key"], source_key="nuclei",
                                        status="present", resolved_path=str(contour), segmentation_id="n1")])
            imported = load_nucleus_qc(config, images, assets, qc.sha256)
            self.assertEqual(len(imported["rounds"]), 24)
            self.assertTrue(imported["objects"].qc_n_comparison_rounds.eq(8).all())
            self.assertTrue(imported["rounds"].loc[imported["rounds"]["round"].isin([7, 8, 9]), "score"].isna().all())
            evidence = pd.read_csv(folder / "round_qc.csv", dtype=str, keep_default_na=False)
            evidence.drop(evidence[evidence["round"] == "7"].index[0]).to_csv(folder / "round_qc.csv", index=False)
            manifest_path = folder / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["files"]["round_qc.csv"] = qc.sha256(folder / "round_qc.csv")
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "evidence is incomplete"):
                load_nucleus_qc(config, images, assets, qc.sha256)


if __name__ == "__main__":
    unittest.main()
