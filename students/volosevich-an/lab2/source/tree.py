from __future__ import annotations
from typing import Tuple
import numpy as np


class Node:

    def __init__(
        self,
        feature: int | None = None,
        threshold: float | None = None,
        left: "Node | None" = None,
        right: "Node | None" = None,
        value: int | None = None,
        left_prob: float = 0.5,
        right_prob: float = 0.5
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.left_prob = left_prob
        self.right_prob = right_prob

    def is_leaf(self) -> bool:
        return self.value is not None


class ID3Tree:

    def __init__(self, max_depth: int = 10, min_samples: int = 5):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.root: Node | None = None

    def gini(self, y: np.ndarray) -> float:
        _, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()
        return float(1 - np.sum(p ** 2))

    def gini_split(self, y_left: np.ndarray, y_right: np.ndarray) -> float:
        n = len(y_left) + len(y_right)
        return (
            (len(y_left) / n) * self.gini(y_left)
            + (len(y_right) / n) * self.gini(y_right)
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.root = self._build(X, y, 0)

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        mask = ~np.isnan(y)
        y_clean = y[mask]
        X_clean = X[mask, :]

        if len(y_clean) == 0:
            return Node(value=0)

        if len(np.unique(y_clean)) == 1:
            return Node(value=int(y_clean[0]))

        if depth >= self.max_depth or len(y_clean) < self.min_samples:
            return Node(value=self._majority(y_clean))

        feature, threshold, left_prob, right_prob = self._best_split(X_clean, y_clean)
        if feature is None:
            return Node(value=self._majority(y_clean))

        col = X_clean[:, feature]
        mask_left = col <= threshold
        mask_right = col > threshold

        left_child = self._build(X_clean[mask_left], y_clean[mask_left], depth + 1)
        right_child = self._build(X_clean[mask_right], y_clean[mask_right], depth + 1)

        return Node(
            feature=feature,
            threshold=threshold,
            left=left_child,
            right=right_child,
            left_prob=left_prob,
            right_prob=right_prob
        )

    def _best_split(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[int | None, float | None, float, float]:

        best_gini = float("inf")
        best_feature = None
        best_threshold = None
        best_left_prob = 0.5
        best_right_prob = 0.5

        n_features = X.shape[1]

        for f in range(n_features):
            col = X[:, f]
            valid = col[~np.isnan(col)]
            if len(valid) == 0:
                continue
            thresholds = np.unique(valid)
            for t in thresholds:
                mask_left = col <= t
                mask_right = col > t
                y_left = y[mask_left]
                y_right = y[mask_right]
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                g = self.gini_split(y_left, y_right)
                if g < best_gini:
                    best_gini = g
                    best_feature = f
                    best_threshold = float(t)
                    total = mask_left.sum() + mask_right.sum()
                    best_left_prob = float(mask_left.sum() / total)
                    best_right_prob = 1.0 - best_left_prob

        return best_feature, best_threshold, best_left_prob, best_right_prob

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_one(x, self.root) for x in X], dtype=int)

    def _predict_one(self, x: np.ndarray, node: Node) -> int:
        if node.is_leaf():
            return int(node.value)

        value = x[node.feature]
        if np.isnan(value):
            if node.left_prob >= node.right_prob:
                return self._predict_one(x, node.left)
            return self._predict_one(x, node.right)

        if value <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def _majority(self, y: np.ndarray) -> int:
        y_clean = y[~np.isnan(y)]
        if len(y_clean) == 0:
            return 0
        values, counts = np.unique(y_clean, return_counts=True)
        return int(values[np.argmax(counts)])

    def prune(self, X_val: np.ndarray, y_val: np.ndarray) -> None:
        self._prune(self.root, X_val, y_val)

    def _prune(self, node: Node | None, X_val: np.ndarray, y_val: np.ndarray) -> None:
        if node is None or node.is_leaf():
            return
        if node.left:
            self._prune(node.left, X_val, y_val)
        if node.right:
            self._prune(node.right, X_val, y_val)
        if node.left and node.right:
            if node.left.is_leaf() and node.right.is_leaf():
                pred_before = self.predict(X_val)
                acc_before = float((pred_before == y_val).mean())
                left_b = node.left
                right_b = node.right
                node.left = None
                node.right = None
                node.value = left_b.value
                pred_after = self.predict(X_val)
                acc_after = float((pred_after == y_val).mean())
                if acc_after < acc_before:
                    node.left = left_b
                    node.right = right_b
                    node.value = None
