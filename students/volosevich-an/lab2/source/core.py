import numpy as np
from tree import ID3Tree


class BaseEnsemble:

    def __init__(self, n_estimators: int):
        self.n_estimators = n_estimators

        self.models = []
        self.bootstrap_idx = []
        self.oob_idx = []

    def _aggregate(self, preds: np.ndarray) -> np.ndarray:
        return np.mean(preds, axis=0)

    def _oob_predict(self, X: np.ndarray):
        n_samples = X.shape[0]

        preds = np.zeros(n_samples, dtype=float)
        counts = np.zeros(n_samples, dtype=int)

        for model, oob in zip(self.models, self.oob_idx):
            if len(oob) == 0:
                continue

            pred = model.predict(X[oob])
            preds[oob] += pred
            counts[oob] += 1

        mask = counts > 0
        preds[mask] /= counts[mask]

        return preds, mask

    def oob_score(self, X: np.ndarray, y: np.ndarray) -> float:
        preds, mask = self._oob_predict(X)
        return float(np.mean(preds[mask] == y[mask]))


class RandomForest(BaseEnsemble):

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples: int = 5,
        max_features: str | int = "sqrt",
        random_state: int = 42,
    ):
        super().__init__(n_estimators)

        self.max_depth = max_depth
        self.min_samples = min_samples
        self.max_features = max_features

        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

        self.feature_subsets = []

    def _get_feature_subset(self, n_features: int):
        if self.max_features == "sqrt":
            k = int(np.sqrt(n_features))
        elif self.max_features == "log2":
            k = int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            k = self.max_features
        else:
            k = n_features

        return self.rng.choice(n_features, size=k, replace=False)

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape

        self.models = []
        self.bootstrap_idx = []
        self.oob_idx = []
        self.feature_subsets = []

        for _ in range(self.n_estimators):
            # bootstrap
            idx = self.rng.integers(0, n_samples, n_samples)
            oob = np.setdiff1d(np.arange(n_samples), idx)

            X_boot = X[idx]
            y_boot = y[idx]

            # random subspace
            features = self._get_feature_subset(n_features)

            tree = ID3Tree(
                max_depth=self.max_depth,
                min_samples=self.min_samples
            )

            tree.fit(X_boot[:, features], y_boot)

            self.models.append(tree)
            self.bootstrap_idx.append(idx)
            self.oob_idx.append(oob)
            self.feature_subsets.append(features)

        return self

    def predict(self, X: np.ndarray):
        preds = []

        for model, features in zip(self.models, self.feature_subsets):
            pred = model.predict(X[:, features])
            preds.append(pred)

        preds = np.array(preds)
        return np.round(self._aggregate(preds)).astype(int)

    def _oob_predict(self, X: np.ndarray):
        n_samples = X.shape[0]

        preds = np.zeros(n_samples, dtype=float)
        counts = np.zeros(n_samples, dtype=int)

        for model, oob, features in zip(self.models, self.oob_idx, self.feature_subsets):
            if len(oob) == 0:
                continue

            pred = model.predict(X[oob][:, features])
            preds[oob] += pred
            counts[oob] += 1

        mask = counts > 0
        preds[mask] /= counts[mask]

        return preds, mask


def compute_oob_importance(model: RandomForest, X: np.ndarray, y: np.ndarray):
    base_score = model.oob_score(X, y)

    n_features = X.shape[1]
    importances = np.zeros(n_features, dtype=float)

    for j in range(n_features):
        X_perm = X.copy()

        for oob in model.oob_idx:
            if len(oob) == 0:
                continue

            perm = np.random.permutation(oob)
            X_perm[oob, j] = X[perm, j]

        score = model.oob_score(X_perm, y)

        importances[j] = (score - base_score) / base_score

    return importances
