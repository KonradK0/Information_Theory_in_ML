import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import kmeans
import gausian_mixture_model

gauss1 = stats.multivariate_normal([0, 0], [[20, 0], [0, 20]])
gauss2 = stats.multivariate_normal([12, 12], [[3, 0], [0, 3]])
gauss3 = stats.multivariate_normal([-12, 12], [[3, 0], [0, 3]])

dataset = []
for _ in range(600):
    dataset.append(gauss1.rvs())
for _ in range(200):
    dataset.append(gauss2.rvs())
for _ in range(200):
    dataset.append(gauss3.rvs())
dataset = np.array(dataset)


kmeans = kmeans.KMeans(X=dataset, K=3)
preds = []
for te_x in dataset:
    idx, center = kmeans.predict(te_x)
    preds.append(idx)
preds = np.array(preds)

plt.subplot(2, 1, 1)
plt.scatter(dataset[:, 0], dataset[:, 1], c=preds, s=50, cmap='viridis', alpha=0.5)
plt.scatter(kmeans.centers[:, 0], kmeans.centers[:, 1], c='black', s=200)
plt.title('KMeans - My implementation')
plt.xticks(visible=False)

kmeans_sklearn = KMeans(n_clusters=3, init='random', algorithm='full')
kmeans_sklearn.fit(dataset)
preds = kmeans_sklearn.predict(dataset)
plt.subplot(2, 1, 2)
plt.scatter(dataset[:, 0], dataset[:, 1], c=preds, s=50, cmap='viridis', alpha=0.5)
plt.scatter(kmeans_sklearn.cluster_centers_[:, 0], kmeans_sklearn.cluster_centers_[:, 1], c='black', s=200)
plt.title('KMeans - Scikit learn')
plt.show()

clusters = np.array([np.array(cluster) for _, cluster in kmeans.clusters.items()])
gmm = gausian_mixture_model.GaussianMixtureModel(dataset, clusters)
gmm.visualize()

