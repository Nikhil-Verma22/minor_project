import os
import sys
import unittest
import numpy as np
import torch
from functools import reduce

sys.path.insert(0, "d:/Minor")
sys.path.insert(0, "d:/Minor/model_repos/SGG-Benchmark")
from sgg_benchmark.data.datasets.evaluation.sgg_metrics import (
    _compute_pred_matches,
    _triplet,
    SGRecall,
    SGMeanRecall
)
from experiments.REACT.ontology_projection import OntologyProjector

class TestEvaluatorSynthetic(unittest.TestCase):
    """
    Synthetic unit tests validating:
    1. Vocabulary sizes for normalized views (6, 8, 20).
    2. Prediction ontology projection and directionality preservation.
    3. Dual empty-GT recall calculation (eligible images vs all-200 images).
    4. Duplicate GT triplet matching prevention via union.
    5. Methodological compatibility guard asserting rejection of cross-dataset SGDet.
    """

    def setUp(self):
        self.projector = OntologyProjector("d:/Minor/evaluation_sets/evaluation_ontology.json")

    def test_vocabulary_sizes(self):
        """Test that strict, extended, and tri_dataset views have exact expected sizes."""
        self.assertEqual(self.projector.get_view_size("strict"), 6)
        self.assertEqual(self.projector.get_view_size("extended"), 8)
        self.assertEqual(self.projector.get_view_size("tri_dataset"), 20)
        
        strict_vocab = self.projector.get_view_vocab("strict")
        self.assertEqual(len(strict_vocab), 6)
        extended_vocab = self.projector.get_view_vocab("extended")
        self.assertEqual(len(extended_vocab), 8)
        tri_vocab = self.projector.get_view_vocab("tri_dataset")
        self.assertEqual(len(tri_vocab), 20)

    def test_compatibility_guards(self):
        """Test that cross-dataset SGDet is rejected, while native is accepted."""
        # Native PSG on PSG -> VALID
        res_psg = self.projector.validate_compatibility("PSG", "PSG", "sgdet")
        self.assertEqual(res_psg["status"], "VALID")
        self.assertEqual(res_psg["exposure_label"], "NATIVE")

        # Native VG150 on VG150 -> VALID
        res_vg = self.projector.validate_compatibility("VG150", "VG150", "sgdet")
        self.assertEqual(res_vg["status"], "VALID")

        # VG150 on IndoorVG -> Flagged PRIOR_TRAINING_EXPOSURE
        res_leak = self.projector.validate_compatibility("VG150", "IndoorVG", "predcls")
        self.assertEqual(res_leak["exposure_label"], "PRIOR_TRAINING_EXPOSURE")

        # Cross-dataset SGDet PSG on VG150 -> MUST RAISE ValueError
        with self.assertRaises(ValueError):
            self.projector.validate_compatibility("PSG", "VG150", "sgdet")

        # Cross-dataset SGDet VG150 on OpenImages -> MUST RAISE ValueError
        with self.assertRaises(ValueError):
            self.projector.validate_compatibility("VG150", "OpenImages", "sgdet")

    def test_directionality_and_projection(self):
        """
        Test that projection onto normalized views preserves directionality
        and correctly maps known predicates while zeroing unmapped predicates.
        """
        # PSG native classes: index 0 is __background__, 4 is 'on'
        source_classes = ["__background__", "over", "in front of", "beside", "on", "sitting on"]
        # In strict view (size 6): ['__background__', 'eating', 'hanging from', 'holding', 'in', 'on', 'wearing']
        target_classes = ["__background__", "eating", "hanging from", "holding", "in", "on", "wearing"]

        # Synthetic prediction: 2 pairs
        # Pair 0: high score on 'on' (index 4)
        # Pair 1: high score on 'over' (index 1 - not in strict)
        pred_scores = torch.zeros((2, len(source_classes)))
        pred_scores[0, 4] = 0.95  # 'on'
        pred_scores[1, 1] = 0.88  # 'over'

        projected = self.projector.project_prediction(
            pred_rel_scores=pred_scores,
            source_dataset="PSG",
            view_name="strict",
            source_ind_to_predicates=source_classes,
            target_ind_to_predicates=target_classes,
        )

        self.assertEqual(projected.shape, (2, 7))
        # Pair 0 should map 'on' to target_classes index 5 ('on')
        self.assertAlmostEqual(projected[0, 5].item(), 0.95, places=4)
        # Pair 1 ('over') is not in strict view -> all non-background scores should be 0.0
        self.assertAlmostEqual(projected[1, 1:].sum().item(), 0.0, places=4)

    def test_duplicate_gt_triplet_matching(self):
        """
        Test that multiple predicted triplets matching the same GT triplet
        only count as 1 hit (via np.union1d) rather than artificially inflating recall.
        """
        # 1 Ground truth triplet: box 0, box 1, predicate 1
        gt_triplets = np.array([[10, 1, 20]])  # subject class 10, predicate 1, object class 20
        gt_boxes = np.array([[0, 0, 10, 10, 20, 20, 30, 30]])  # sub box (0,0,10,10), obj box (20,20,30,30)

        # 3 Predictions: all 3 predict the exact same triplet with identical overlapping boxes
        pred_triplets = np.array([
            [10, 1, 20],
            [10, 1, 20],
            [10, 1, 20],
        ])
        pred_boxes = np.array([
            [0, 0, 10, 10, 20, 20, 30, 30],
            [0, 0, 10, 10, 20, 20, 30, 30],
            [0, 0, 10, 10, 20, 20, 30, 30],
        ])

        pred_to_gt = _compute_pred_matches(gt_triplets, pred_triplets, gt_boxes, pred_boxes, iou_thres=0.5)
        self.assertEqual(len(pred_to_gt), 3)
        # Each prediction matched GT 0
        for m in pred_to_gt:
            self.assertEqual(m, [0])

        # Test union at K=3: must result in exactly 1 hit
        match = reduce(np.union1d, pred_to_gt[:3])
        self.assertEqual(len(match), 1)
        self.assertEqual(match[0], 0)

    def test_dual_empty_gt_recall_policy(self):
        """
        Test dual empty-ground-truth policy:
        Image 1: 1 GT, 1 Hit -> Recall = 1.0
        Image 2: 0 GT (empty) -> Excluded from eligible mean, but assigned 0.0 in all-image sensitivity
        Eligible R@20: 1.0 / 1 = 1.0
        All-image R@20: (1.0 + 0.0) / 2 = 0.5
        """
        result_dict = {}
        evaluator = SGRecall(result_dict, name='R', key='recall', k_list=[20])
        evaluator.register_container('sgdet')

        # Image 1: has GT and hit
        local_c1 = {
            'pred_to_gt': [[0]],
            'gt_rels': np.array([[0, 1, 1]])
        }
        evaluator.calculate({}, local_c1, 'sgdet')

        eligible_recall = np.mean(result_dict['sgdet_recall'][20])
        self.assertAlmostEqual(eligible_recall, 1.0, places=4)

        # In dual policy, for all-image sensitivity, an empty image gets 0.0
        all_image_recalls = list(result_dict['sgdet_recall'][20]) + [0.0]
        sensitivity_recall = np.mean(all_image_recalls)
        self.assertAlmostEqual(sensitivity_recall, 0.5, places=4)

if __name__ == '__main__':
    unittest.main()
