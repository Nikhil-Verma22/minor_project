import json
import os
from typing import Dict, List, Optional, Tuple
import torch
import numpy as np

class OntologyProjector:
    """
    Dynamic Prediction Ontology Projection Layer.
    Projects raw model predictions from native source dataset predicates onto
    normalized evaluation views (strict, extended, tri_dataset) using evaluation_ontology.json.
    Strictly preserves subject and object endpoint directionality and filters unmapped relations.
    """

    def __init__(self, ontology_path: str = "d:/Minor/evaluation_sets/evaluation_ontology.json"):
        if not os.path.exists(ontology_path):
            raise FileNotFoundError(f"Ontology file not found at: {ontology_path}")
        with open(ontology_path, "r", encoding="utf-8") as f:
            self.ontology = json.load(f)

        self.vocabularies = self.ontology.get("vocabularies", {})
        self.dataset_normalizations = self.ontology.get("dataset_normalizations", {})

        # Define canonical view names
        self.view_key_map = {
            "strict": "strict_core_4way",
            "strict_core_4way": "strict_core_4way",
            "extended": "extended_core_4way",
            "extended_core_4way": "extended_core_4way",
            "tri_dataset": "tri_dataset_core",
            "tri_dataset_core": "tri_dataset_core",
        }

    def get_view_vocab(self, view_name: str) -> Dict[str, int]:
        """Returns the dictionary mapping predicate name -> 1-indexed ontology ID for the view."""
        key = self.view_key_map.get(view_name)
        if not key or key not in self.vocabularies:
            raise ValueError(f"Unknown view: {view_name}. Available: strict, extended, tri_dataset")
        return self.vocabularies[key]["ontology_ids"]

    def get_view_size(self, view_name: str) -> int:
        """Returns vocabulary size (number of relationship predicates) for the view."""
        key = self.view_key_map.get(view_name)
        if not key or key not in self.vocabularies:
            raise ValueError(f"Unknown view: {view_name}")
        return self.vocabularies[key]["size"]

    def validate_compatibility(self, model_dataset: str, eval_dataset: str, task: str) -> Dict[str, str]:
        """
        Enforces architectural and methodological compatibility guards:
        - Cross-dataset SGDet without object alignment is rejected.
        - VG150 model on IndoorVG is flagged as PRIOR_TRAINING_EXPOSURE.
        """
        # Normalize dataset names
        m_ds = model_dataset.upper().replace(" ", "").replace("_", "")
        e_ds = eval_dataset.upper().replace(" ", "").replace("_", "")

        # Check exposure
        exposure = "NATIVE"
        if m_ds == e_ds:
            exposure = "NATIVE"
        elif "VG150" in m_ds and "INDOORVG" in e_ds:
            exposure = "PRIOR_TRAINING_EXPOSURE"
        else:
            exposure = "CROSS_DATASET"

        if task.lower() == "sgdet" and m_ds != e_ds:
            raise ValueError(
                f"Cross-dataset SGDet is METHODOLOGICALLY INVALID! Model dataset '{model_dataset}' and "
                f"evaluation dataset '{eval_dataset}' have disjoint object vocabularies without an object ontology mapping."
            )

        return {
            "status": "VALID",
            "task": task.lower(),
            "source_dataset": model_dataset,
            "eval_dataset": eval_dataset,
            "exposure_label": exposure,
        }

    def project_prediction(
        self,
        pred_rel_scores: torch.Tensor,
        source_dataset: str,
        view_name: str,
        source_ind_to_predicates: List[str],
        target_ind_to_predicates: List[str],
    ) -> torch.Tensor:
        """
        Projects predicted relationship logits/scores to the target ontology view.
        pred_rel_scores: [N_pairs, num_source_classes] (where index 0 is __background__)
        Returns: [N_pairs, num_target_classes] (where index 0 is __background__)
        """
        if view_name in ("native", "none", None):
            return pred_rel_scores

        view_key = self.view_key_map.get(view_name)
        if not view_key:
            raise ValueError(f"Unknown view: {view_name}")

        flag_name = f"in_{view_name}"
        if view_name == "strict":
            flag_name = "in_strict_4way"
        elif view_name == "extended":
            flag_name = "in_extended_4way"
        elif view_name == "tri_dataset":
            flag_name = "in_tri_dataset"

        ds_norm = self.dataset_normalizations.get(source_dataset)
        if not ds_norm:
            raise ValueError(f"Dataset normalization mapping for '{source_dataset}' not found in ontology!")

        mapping = ds_norm.get("source_to_ontology_mapping", {})
        num_target = len(target_ind_to_predicates)
        device = pred_rel_scores.device
        dtype = pred_rel_scores.dtype
        n_pairs = pred_rel_scores.shape[0]

        # Initialize projected scores tensor
        projected = torch.zeros((n_pairs, num_target), device=device, dtype=dtype)
        # Background score remains background
        if pred_rel_scores.shape[1] > 0:
            projected[:, 0] = pred_rel_scores[:, 0]

        # Map each source predicate
        for s_idx in range(1, len(source_ind_to_predicates)):
            if s_idx >= pred_rel_scores.shape[1]:
                break
            src_name = source_ind_to_predicates[s_idx]
            info = mapping.get(src_name)
            if not info:
                continue

            # Check if this predicate is included in the target view
            if not info.get(flag_name, False):
                continue

            canonical_name = info.get("normalized_canonical_predicate")
            if not canonical_name:
                continue

            if canonical_name in target_ind_to_predicates:
                t_idx = target_ind_to_predicates.index(canonical_name)
                # If multiple source predicates map to the same target, take the maximum
                src_col = pred_rel_scores[:, s_idx]
                projected[:, t_idx] = torch.maximum(projected[:, t_idx], src_col)

        return projected
