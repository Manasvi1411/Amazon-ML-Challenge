"""Model training, macro F_0.5 metric evaluation, and threshold calibration."""

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

def calculate_entity_f05(predicted_set: Set[str], ground_truth_set: Set[str]) -> float:
    """
    Calculate F_0.5 score for a single Source 1 entity according to challenge rules.
    - If ground truth is empty (singleton):
        - Score is 1.0 if predicted is empty
        - Score is 0.0 if any match is predicted
    - If ground truth is not empty:
        - If predicted is empty: score is 0.0
        - Otherwise: Precision = |P ∩ T| / |P|, Recall = |P ∩ T| / |T|
          F_0.5 = (1.25 * P * R) / (0.25 * P + R)
    """
    # Singleton check
    if not ground_truth_set:
        return 1.0 if not predicted_set else 0.0

    # Non-singleton but model predicted nothing
    if not predicted_set:
        return 0.0

    inter = len(predicted_set.intersection(ground_truth_set))
    if inter == 0:
        return 0.0

    precision = inter / len(predicted_set)
    recall = inter / len(ground_truth_set)

    denom = (0.25 * precision) + recall
    if denom == 0.0:
        return 0.0

    return (1.25 * precision * recall) / denom

def calculate_macro_f05(
    predictions_map: Dict[str, Set[str]],
    ground_truth_map: Dict[str, Set[str]],
    all_s1_ids: List[str],
) -> Tuple[float, float, float]:
    """
    Compute macro-averaged F_0.5, macro-precision, and macro-recall across all S1 entities.
    """
    scores = []
    precisions = []
    recalls = []

    for s1_id in all_s1_ids:
        pred_set = predictions_map.get(s1_id, set())
        gt_set = ground_truth_map.get(s1_id, set())

        f05 = calculate_entity_f05(pred_set, gt_set)
        scores.append(f05)

        if not gt_set:
            if not pred_set:
                precisions.append(1.0)
                recalls.append(1.0)
            else:
                precisions.append(0.0)
                recalls.append(1.0)
        else:
            if not pred_set:
                precisions.append(0.0)
                recalls.append(0.0)
            else:
                inter = len(pred_set.intersection(gt_set))
                precisions.append(inter / len(pred_set))
                recalls.append(inter / len(gt_set))

    return float(np.mean(scores)), float(np.mean(precisions)), float(np.mean(recalls))

class EntityMatchingModel:
    """LightGBM-based entity matching model with macro F_0.5 threshold optimization."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model: Optional[lgb.LGBMClassifier] = None
        self.optimal_threshold: float = 0.65
        self.feature_names: List[str] = []

    def train_and_calibrate(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        pair_list: List[Tuple[str, str]],
        ground_truth_map: Dict[str, Set[str]],
        all_s1_ids: List[str],
        n_splits: int = 5,
    ) -> float:
        """
        Train LightGBM model and calibrate optimal probability threshold
        maximizing macro F_0.5 score via out-of-fold predictions.
        """
        self.feature_names = list(X.columns)

        # Set up Out-Of-Fold prediction collector
        unique_s1 = np.array(all_s1_ids)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        
        oof_preds: Dict[Tuple[str, str], float] = {}

        # S1 entity to index mapping in pair_list
        s1_to_pair_indices = defaultdict(list)
        for idx, (s1_id, _) in enumerate(pair_list):
            s1_to_pair_indices[s1_id].append(idx)

        for fold, (train_s1_idx, val_s1_idx) in enumerate(kf.split(unique_s1)):
            train_s1 = set(unique_s1[train_s1_idx])
            val_s1 = set(unique_s1[val_s1_idx])

            train_indices = [
                idx for s1 in train_s1 for idx in s1_to_pair_indices.get(s1, [])
            ]
            val_indices = [
                idx for s1 in val_s1 for idx in s1_to_pair_indices.get(s1, [])
            ]

            if not train_indices or not val_indices:
                continue

            X_tr, y_tr = X.iloc[train_indices], y[train_indices]
            X_val = X.iloc[val_indices]

            fold_model = lgb.LGBMClassifier(
                n_estimators=300,
                learning_rate=0.04,
                num_leaves=31,
                min_child_samples=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state + fold,
                verbosity=-1,
                n_jobs=-1,
            )
            fold_model.fit(X_tr, y_tr)
            val_probs = fold_model.predict_proba(X_val)[:, 1]

            for idx, prob in zip(val_indices, val_probs):
                oof_preds[pair_list[idx]] = float(prob)

        # Threshold search over OOF predictions
        best_threshold = 0.65
        best_f05 = -1.0

        threshold_grid = np.linspace(0.40, 0.95, 56)

        for thresh in threshold_grid:
            pred_map: Dict[str, Set[str]] = defaultdict(set)
            for (s1_id, cand_id), prob in oof_preds.items():
                if prob >= thresh:
                    pred_map[s1_id].add(cand_id)

            f05, prec, rec = calculate_macro_f05(pred_map, ground_truth_map, all_s1_ids)
            if f05 > best_f05:
                best_f05 = f05
                best_threshold = float(thresh)

        self.optimal_threshold = best_threshold

        # Fit final model on all data
        self.model = lgb.LGBMClassifier(
            n_estimators=350,
            learning_rate=0.04,
            num_leaves=31,
            min_child_samples=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            verbosity=-1,
            n_jobs=-1,
        )
        self.model.fit(X, y)

        return best_f05

    def predict_matches(
        self,
        X_test: pd.DataFrame,
        pair_list: List[Tuple[str, str]],
        threshold: Optional[float] = None,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Run inference over candidate pairs.
        Returns dict mapping:
            s1_id -> list of (cand_id, probability) that exceed threshold
        """
        if self.model is None or X_test.empty or not pair_list:
            return {}

        thresh = threshold if threshold is not None else self.optimal_threshold
        probs = self.model.predict_proba(X_test[self.feature_names])[:, 1]

        matches_map: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for (s1_id, cand_id), prob in zip(pair_list, probs):
            if prob >= thresh:
                matches_map[s1_id].append((cand_id, float(prob)))

        # Sort each list by probability descending
        for s1_id in matches_map:
            matches_map[s1_id].sort(key=lambda x: x[1], reverse=True)

        return matches_map
