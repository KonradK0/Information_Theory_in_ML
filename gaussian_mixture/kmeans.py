import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


class KMeans:
    def __init__(self, K, X):
        self.K = K
        self.X = X
        self.centers = self._initialize_centers(X)
        self.clusters = defaultdict(list)
        self._fit(X)

    def _initialize_centers(self, X):
        idx = np.random.choice(list(range(len(X))), size=self.K, replace=False)
        return X[idx].copy()

    def _fit(self, X, max_iter=100):
        for _ in range(max_iter):
            clusters = defaultdict(list)
            for i, _ in enumerate(self.centers):
                for j, x in enumerate(X):
                    idx, _ = self.predict(x)
                    clusters[idx].append(x)
            new_centers = []
            for _, cluster in clusters.items():
                new_centers.append(np.mean(cluster, axis=0))
            new_centers = np.array(new_centers)
            if np.all(np.isclose(new_centers, self.centers)):
                self.clusters = clusters
                break
            self.centers = new_centers

    def predict(self, x):
        idx = np.argmin(np.linalg.norm(x - self.centers, axis=1))
        return idx, self.centers[idx]

    def visualize(self, preds):
        plt.scatter(self.X[:, 0], self.X[:, 1], c=preds, s=50, cmap='viridis', alpha=0.5)
        plt.scatter(self.centers[:, 0], self.centers[:, 1], c='black', s=200)
        plt.show()
