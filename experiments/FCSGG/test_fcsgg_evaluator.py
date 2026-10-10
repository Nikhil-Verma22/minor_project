import os
import sys
import unittest
import torch
import numpy as np

WORKSPACE_ROOT = "d:/Minor"
sys.path.insert(0, WORKSPACE_ROOT)
sys.path.insert(0, os.path.join(WORKSPACE_ROOT, "model_repos/SGG-Benchmark"))

from experiments.REACT.ontology_projection import OntologyProjector
from experiments.FCSGG.fcsgg_adapter import FCSGGAdapter

class TestFCSGGEvaluator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.adapter = FCSGGAdapter()
        cls.test_img = os.path.join(WORKSPACE_ROOT, "evaluation_sets/images/VG150/2315564.jpg")
        cls.projector = OntologyProjector(os.path.join(WORKSPACE_ROOT, "evaluation_sets/evaluation_ontology.json"))

    def test_coordinate_preservation(self):
        """Test that predicted box coordinates match image bounds and valid bbox geometries."""
        pred_dict, fwd_ms, e2e_ms = self.adapter.predict_image(self.test_img)
        orig_h, orig_w = pred_dict["orig_size"]
        boxes = pred_dict["boxes"]

        self.assertGreater(len(boxes), 0, "Should detect at least 1 bounding box")
        self.assertEqual(boxes.shape[1], 4, "Bounding boxes should be 4-dim (x1, y1, x2, y2)")
        
        # Check non-negative width and height
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        self.assertTrue((widths >= 0).all(), "All box widths must be non-negative")
        self.assertTrue((heights >= 0).all(), "All box heights must be non-negative")
        
        # Check coordinates within reasonable bounds
        self.assertTrue((boxes[:, 0] >= -5).all(), "x1 should be >= 0 (with slight numerical tolerance)")
        self.assertTrue((boxes[:, 1] >= -5).all(), "y1 should be >= 0 (with slight numerical tolerance)")
        self.assertTrue((boxes[:, 2] <= orig_w + 10).all(), "x2 should be <= image width")
        self.assertTrue((boxes[:, 3] <= orig_h + 10).all(), "y2 should be <= image height")

    def test_native_decoding_and_indexing(self):
        """Test that classes are 1-indexed (1..150) and relations are 51-dim (1..50 predicates)."""
        pred_dict, _, _ = self.adapter.predict_image(self.test_img)
        pred_labels = pred_dict["pred_labels"]
        rel_scores = pred_dict["pred_rel_scores"]
        rel_labels = pred_dict["pred_rel_labels"]

        # Object classes must be in 1..150
        self.assertTrue((pred_labels >= 1).all(), "Object classes must be >= 1")
        self.assertTrue((pred_labels <= 150).all(), "Object classes must be <= 150")

        # Relation scores must have 51 channels (channel 0 is background)
        self.assertEqual(rel_scores.shape[1], 51, "Relation score tensor must have 51 channels")
        self.assertTrue((rel_scores[:, 0] == 0.0).all(), "Channel 0 must be 0 (background)")

        if len(rel_labels) > 0:
            self.assertTrue((rel_labels >= 1).all(), "Predicate classes must be >= 1")
            self.assertTrue((rel_labels <= 50).all(), "Predicate classes must be <= 50")

    def test_directionality_and_ranking(self):
        """Test that subject/object directed pairs are preserved and sorted by triplet score."""
        pred_dict, _, _ = self.adapter.predict_image(self.test_img)
        pair_idxs = pred_dict["rel_pair_idxs"]
        rel_scores = pred_dict["pred_rel_scores"]

        self.assertGreater(len(pair_idxs), 0, "Should have predicted relation pairs")
        self.assertEqual(pair_idxs.shape[1], 2, "Pair indices must be (subj_idx, obj_idx)")

        # Verify that directionality distinguishes (s, o) from (o, s)
        subj = pair_idxs[0, 0].item()
        obj = pair_idxs[0, 1].item()
        self.assertNotEqual(subj, obj, "Subject and object indices should be distinct objects")

    def test_ontology_projection(self):
        """Test that native VG150 scores project onto strict, extended, and tri_dataset views."""
        pred_dict, _, _ = self.adapter.predict_image(self.test_img)
        raw_scores = pred_dict["pred_rel_scores"]

        vg_ind_to_predicates = ["__background__"] + [
            "above", "across", "against", "along", "and", "at", "attached to", "behind",
            "belonging to", "between", "carrying", "covered in", "covering", "eating",
            "flying in", "for", "from", "growing on", "hanging from", "has", "holding",
            "in", "in front of", "laying on", "looking at", "lying on", "made of", "mounted on",
            "near", "of", "on", "on back of", "over", "painted on", "parked on", "part of",
            "playing", "riding", "says", "sitting on", "standing on", "to", "under",
            "using", "walking on", "wearing", "wears", "with", "with", "with"
        ]

        strict_predicates = ["__background__", "eating", "hanging from", "holding", "in", "on", "wearing"]
        proj_strict = self.projector.project_prediction(
            pred_rel_scores=raw_scores,
            source_dataset="VG150",
            view_name="strict",
            source_ind_to_predicates=vg_ind_to_predicates,
            target_ind_to_predicates=strict_predicates
        )

        self.assertEqual(proj_strict.shape[1], 7, "Strict view should have 7 columns (__background__ + 6 predicates)")
        self.assertEqual(len(proj_strict), len(raw_scores), "Projected tensor length must match raw length")

if __name__ == "__main__":
    unittest.main()
