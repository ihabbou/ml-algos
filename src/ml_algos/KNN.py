
import numpy as np


class KNN:
    """K Nearest Neighbours Classifier. 
    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    """

    def __init__(self, k=3) -> None:
        self.k = k

    def fit(self, X, y):
        ""
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        ""
        import numpy as np
        y_pred = [self._predict_one(x) for x in X]

        return np.array(y_pred)

    def _predict_one(self, x):
        ""
        from ml_algos.utils import distance
        from typing import Counter
        # Calculate distance to all points
        distances = [distance(x, xt) for xt in self.X_train]
        # Get labels of nearest k points
        nearest_labels = zip(distances, list(self.y_train))
        nearest_labels = sorted(nearest_labels, key=lambda x: x[0])
        nearest_labels_k = [x[1] for x in nearest_labels][:self.k]
        # Majority vote
        return Counter(nearest_labels_k).most_common(1)[0][0]
