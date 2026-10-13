"""Check paired-round denominators, missing scores, and shared patch rejection."""

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_dapi_rounds import average_ranks, comparison_data, correlation, shared_retention_data
from plan_scoring import create_plan


def image(patch, r, score=None, status="ok", well="well_a"):
    return {"well_slug": well, "patch_id": patch,
            "acquisition_id": f"measurement_{r:02d}_dapi", "measurement": str(r),
            "image_score_status": status, "max_duplicate_ncc_excess": score}


class RoundComparisonTests(unittest.TestCase):
    def test_tied_ranks_and_undefined_correlations(self):
        np.testing.assert_array_equal(average_ranks(np.array([1., 1., 2.])), [1.5, 1.5, 3.])
        self.assertAlmostEqual(correlation([1, 1, 2], [1, 2, 2], 3, rank=True), .5)
        self.assertAlmostEqual(correlation([1, 2, 3], [3, 2, 1], 3), -1)
        self.assertIsNone(correlation([1, 1, 1], [1, 2, 3], 3))
        self.assertIsNone(correlation([1, 2], [1, 2], 3))

    def test_directional_overlap_excludes_unknown_but_counts_no_source_seam(self):
        rows = [image("a", 1, .8), image("a", 2, .8),
                image("b", 1, .6), image("b", 2, .1),
                image("c", 1, .1), image("c", 2, .8),
                image("d", 1, status="no_source_seam"),
                image("e", 1, .9), image("e", 2, status="no_evaluable_seam"),
                image("f", 1, .9), image("f", 2, status="no_source_seam")]
        data = comparison_data(rows, [.5])
        lookup = {(row["source_round"], row["target_round"]): row for row in data["overlaps"]}
        forward, reverse = lookup[1, 2], lookup[2, 1]
        self.assertEqual(forward["paired_assessable_patches"], 4)
        self.assertEqual(forward["matched_but_unassessable_patches"], 1)
        self.assertEqual(forward["source_flagged_in_pair"], 3)
        self.assertEqual(forward["both_flagged"], 1)
        self.assertAlmostEqual(forward["probability_target_flagged_given_source"], 1 / 3)
        self.assertAlmostEqual(reverse["probability_target_flagged_given_source"], 1 / 2)
        correlations = {(row["source_round"], row["target_round"]): row for row in data["correlations"]}
        self.assertEqual(correlations[1, 2]["paired_evaluable_patches"], 3)
        self.assertEqual(correlations[1, 3]["paired_evaluable_patches"], 0)
        self.assertIsNone(correlations[1, 3]["pearson_r"])

    def test_one_round_discards_all_twelve_images_at_cutoff_equality(self):
        rows = [image("a", r, .8 if r == 9 else .1) for r in range(1, 13)]
        rows += [image("b", r, .2) for r in range(1, 13)]
        rows += [image("c", r, status="no_source_seam") for r in range(1, 13)]
        data = comparison_data(rows, [.8])
        curve = next(row for row in shared_retention_data(data["patches"], [.8]) if row["cutoff"] == .8)
        self.assertEqual(curve["spatial_patches"], 3)
        self.assertEqual(curve["available_dapi_images"], 36)
        self.assertEqual(curve["patches_with_all_12_dapi_images"], 3)
        self.assertEqual(curve["patches_rejected_in_all_rounds"], 1)
        self.assertEqual(curve["dapi_images_discarded_by_shared_filter"], 12)
        self.assertEqual(curve["dapi_images_flagged_independently"], 1)
        self.assertEqual(curve["additional_dapi_images_discarded_for_consistency"], 11)
        self.assertAlmostEqual(curve["fraction_spatial_patches_kept"], 2 / 3)
        self.assertAlmostEqual(curve["fraction_dapi_images_kept_by_shared_filter"], 2 / 3)
        self.assertEqual(curve["patches_without_evaluable_dapi_score"], 1)
        flag = next(row for row in data["shared_masks"] if row["patch_id"] == "a")
        self.assertTrue(flag["reject_all_rounds"])
        self.assertEqual(flag["triggering_dapi_rounds"], "9")

    def test_patch_identity_includes_well_and_duplicate_images_fail(self):
        rows = [image("a", 1, .8), image("a", 1, .1, well="well_b")]
        self.assertEqual(len(comparison_data(rows, [.5])["patches"]), 2)
        with self.assertRaises(ValueError):
            comparison_data([rows[0], rows[0]], [.5])

    def test_balanced_plan_keeps_twelve_dapi_rounds_at_every_location(self):
        with tempfile.TemporaryDirectory(dir="/private/tmp") as tmp:
            root = Path(tmp)
            acquisitions = [f"measurement_{r:02d}_dapi" for r in range(1, 13)] + ["measurement_01_488_emyhc"]
            for plate in ("plate_a", "plate_b"):
                for well in ("well_1", "well_2"):
                    directory = root / "manifests/wells" / f"{plate}_{well}"
                    directory.mkdir(parents=True)
                    with (directory / "patches.csv").open("w", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=["acquisition_id", "plate_name", "patch_id"])
                        writer.writeheader()
                        writer.writerows({"acquisition_id": a, "plate_name": plate, "patch_id": f"patch_{p}"}
                                         for a in acquisitions for p in range(4))
            cache = root / "cache.json"
            cache.write_text(json.dumps({"patch_run": str(root), "completed_wells": 4}))
            plan = create_plan(cache, 4, "patch", "dapi", 42, 3)
            self.assertEqual(plan["acquisition_ids"], acquisitions[:-1])
            self.assertEqual(plan["selected_patch_count"], 4)
            self.assertEqual(plan["selected_image_count"], 48)
            self.assertEqual(plan["scoring_workers"] + plan["gallery_jobs"], 3)
            self.assertTrue(all(entry["patch_count"] == 1 and entry["image_count"] == 12 and entry["image_keys"] is None
                                for entry in plan["wells"].values()))


if __name__ == "__main__":
    unittest.main()
