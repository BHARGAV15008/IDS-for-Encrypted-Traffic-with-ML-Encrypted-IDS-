from typing import List, Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier


class EnsembleClassifier:

    def __init__(self, base_dim: int, num_classes: int, *,
                 n_estimators: int = 80,
                 max_depth: Optional[int] = 16,
                 max_features: str = 'sqrt',
                 n_jobs: int = 1,
                 random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            n_jobs=n_jobs,
            random_state=random_state,
            class_weight=None,
            min_samples_split=4,
            min_samples_leaf=2,
            bootstrap=True
        )
        self.base_dim = base_dim
        self.num_classes = num_classes

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:

        self.model.fit(features, labels)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:

        return self.model.predict_proba(features)

    def predict(self, features: np.ndarray) -> np.ndarray:

        return self.model.predict(features)


