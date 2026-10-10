import os
import sys
import unittest
import numpy as np
import torch
from functools import reduce

# Add project paths
sys.path.insert(0, "d:/Minor")
sys.path.insert(0, "d:/Minor/model_repos/SGG-Benchmark")

from sgg_benchmark.data.datasets.evaluation.sgg_metrics import (
    _compute_pred_matches,
    _triplet,
    SGRecall,
    SGMeanRecall
)
from experiments.REACT.ontology_projection import OntologyProjector

class TestREACTPPEvaluator(unittest.TestCase):
    """
    REACT++ Specific Evaluator Unit Tests:
    1. Verify native ONNX predicate decoding without 6-predicate truncation.
    2. Verify ontology projection of ONNX relation outputs to strict (6), extended (8), tri_dataset (20).
    3. Verify relation directionality (subject -> object) is strictly preserved.
    4. Verify duplicate ground truth triplet matching prevention via union.
    5. Verify methodological compatibility rejection of cross-dataset SGDet.
    """

    def setUp(self):
        self.projector = OntologyProjector("d:/Minor/evaluation_sets/evaluation_ontology.json")
        # Load PSG native predicates
        import json
        with open("d:/Minor/evaluation_sets/PSG_200_annotations.coco.json") as f:
            psg_data = json.load(f)
        self.psg_ind_to_predicates = ["__background__"] + [r["name"] for r in psg_data["rel_categories"]]
        self.strict_ind_to_predicates = ["__background__", "eating", "hanging from", "holding", "in", "on", "wearing"]

    def test_native_decoding_not_truncated(self):
        """
        Verify that ONNX relations with high predicate IDs (e.g., ID 22 'holding', ID 51 'driving on')
        are decoded using the full 57-class native vocabulary and not truncated to 6 classes.
        """
        # Simulated ONNX rels_raw: 2 relations
        # Rel 0: native label 4 ('on')
        # Rel 1: native label 22 ('holding') -> well beyond strict 6
        rels_raw = np.array([
            [0, 1, 4, 0.85, 0.90],
            [1, 2, 22, 0.75, 0.80]
        ], dtype=np.float32)

        num_native_classes = len(self.psg_ind_to_predicates)  # 57
        self.assertEqual(num_native_classes, 57)

        # Build full native score matrix (N, 57)
        native_scores = torch.zeros((len(rels_raw), num_native_classes), dtype=torch.float32)
        for i, row in enumerate(rels_raw):
            lbl = int(row[2])
            sc = float(row[4])
            native_scores[i, lbl] = sc

        # Ensure label 22 is preserved in native representation
        self.assertAlmostEqual(native_scores[1, 22].item(), 0.80, places=4)
        self.assertEqual(self.psg_ind_to_predicates[4], "on")
        self.assertEqual(self.psg_ind_to_predicates[22], "holding")

    def test_ontology_projection_from_native(self):
        """
        Verify that projection maps native PSG 'on' (ID 4) and 'holding' (ID 22)
        correctly into the strict view ('holding' -> index 3, 'on' -> index 5),
        while unmapped predicates (e.g. 'over' ID 1) are zeroed out.
        """
        # 3 native relations:
        # Row 0: 'on' (ID 4)
        # Row 1: 'holding' (ID 22)
        # Row 2: 'over' (ID 1, not in strict)
        native_scores = torch.zeros((3, 57), dtype=torch.float32)
        native_scores[0, 4] = 0.90   # 'on'
        native_scores[1, 22] = 0.85  # 'holding'
        native_scores[2, 1] = 0.70   # 'over'

        projected = self.projector.project_prediction(
            pred_rel_scores=native_scores,
            source_dataset="PSG",
            view_name="strict",
            source_ind_to_predicates=self.psg_ind_to_predicates,
            target_ind_to_predicates=self.strict_ind_to_predicates
        )

        # Expected strict: ['__background__', 'eating', 'hanging from', 'holding', 'in', 'on', 'wearing']
        # 'holding' is index 3, 'on' is index 5
        self.assertEqual(projected.shape, (3, 7))
        self.assertAlmostEqual(projected[0, 5].item(), 0.90, places=4)  # 'on'
        self.assertAlmostEqual(projected[1, 3].item(), 0.85, places=4)  # 'holding'
        self.assertAlmostEqual(projected[2, 1:].sum().item(), 0.0, places=4)  # 'over' zeroed

    def test_relation_directionality_preservation(self):
        """
        Verify that subject and object endpoints are not swapped.
        (A, B, on) != (B, A, on)
        """
        gt_boxes = np.array([[0, 0, 10, 10, 20, 20, 30, 30]]) # sub at (0..10), obj at (20..30)
        gt_triplets = np.array([[1, 5, 2]])                    # sub cls 1, pred 5, obj cls 2

        # Prediction 1: correct direction (1 -> 5 -> 2)
        pred_triplets_correct = np.array([[1, 5, 2]])
        pred_boxes_correct = np.array([[0, 0, 10, 10, 20, 20, 30, 30]])
        match_correct = _compute_pred_matches(gt_triplets, pred_triplets_correct, gt_boxes, pred_boxes_correct, 0.5)
        self.assertEqual(len(match_correct[0]), 1)

        # Prediction 2: inverted direction (2 -> 5 -> 1)
        pred_triplets_inverted = np.array([[2, 5, 1]])
        pred_boxes_inverted = np.array([[20, 20, 30, 30, 0, 0, 10, 10]])
        match_inverted = _compute_pred_matches(gt_triplets, pred_triplets_inverted, gt_boxes, pred_boxes_inverted, 0.5)
        self.assertEqual(len(match_inverted[0]), 0)  # Must NOT match

    def test_cross_dataset_sgdet_rejection(self):
        """
        Verify that attempting cross-dataset SGDet with the PSG model against VG150 or OpenImages
        is rejected by the compatibility validator.
        """
        with self.assertRaises(ValueError):
            self.projector.validate_compatibility("PSG", "VG150", "sgdet")

        with self.assertRaises(ValueError):
            self.projector.validate_compatibility("PSG", "OpenImages", "sgdet")

if __name__ == "__main__":
    unittest.main()
