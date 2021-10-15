import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


class GaussianMixtureModel:

    def __init__(self, X, clusters, k=3):
        self.X = X
        self.n = X.shape[0]
        self.dim = X.shape[1]
        self.k = k
        means = []
        sigmas = np.empty((self.k, self.dim, self.dim))
        pis = []
        for cluster in clusters:
            mean = np.mean(cluster, axis=0)
            means.append(mean)
            pi = len(cluster) / self.n
            pis.append(pi)
        self.means = np.array(means)
        self.pis = np.array(pis)
        for i, cluster in enumerate(clusters):
            diff = cluster - self.means[i]
            sigmas[i] = diff.T @ (diff) / len(cluster)
        self.sigmas = np.array(sigmas)
        self.r = np.zeros((self.n, self.k))
        self.preds = np.full(self.n, -1)
        self._fit(X)

    def _fit(self, X, max_iter=100):
        for _ in range(max_iter):
            pdfs = [stats.multivariate_normal(mean=self.means[i], cov=self.sigmas[i]) for i in range(self.k)]
            for i in range(self.n):
                for j in range(self.k):
                    self.r[i, j] = self.pis[j] * pdfs[j].pdf(X[i])
            self.r /= np.sum(self.r, axis=1).reshape(-1, 1)
            ns = np.sum(self.r, axis=0)
            means = (self.r.T @ X) / ns.reshape(-1, 1)
            sigmas = np.empty((self.k, self.dim, self.dim))
            for i in range(self.k):
                sigmas[i] = np.zeros((self.dim, self.dim))

                for j in range(self.n):
                    diff = X[j] - self.means[i]
                    sigmas[i] += self.r[j, i] * diff.reshape(-1, 1) @ diff.reshape(1, -1)

                sigmas[i] /= ns[i]
            self.pis = ns / self.n
            if np.all(np.isclose(means, self.means)):
                self.preds = np.argmax(self.r, axis=1)
                break
            self.means = means
            self.sigmas = sigmas

    def predict(self):
        return self.preds

    def visualize(self):
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.preds, s=50, cmap='viridis', alpha=0.5)
        plt.scatter(self.means[:, 0], self.means[:, 1], c='black', s=200)
        plt.show()

# a = np.array([[0, 1], [1, 0]])
# sub = np.array([0, 1])
# print(a - sub)
