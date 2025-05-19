import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

#  para se gerar dados sintéticos com 3 clusters
X, y_true = make_blobs(n_samples=150, centers=3, cluster_std=0.6, random_state=42) # para se gerar dados sintétitcos com 3 clusters

# se caso os dados tiverem mais de 2 dimensões, isso aqui é interessante para se reduzir a dimensionalidade com o PCA
pca = PCA(n_components=2) 
X_pca = pca.fit_transform(X)

# aplicação do algoritmo de facto
kmeans = KMeans(n_clusters=3, random_state=42) 
kmeans.fit(X_pca)
labels = kmeans.predict(X_pca)
centroids = kmeans.cluster_centers_

# plot do gráfico.png com clusters encontrados
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, s=50, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.5, marker='X')
plt.title("todos clusters encontrados pelo K-means com PCA")
plt.xlabel("comp. principal I")
plt.ylabel("comp. principal II")
plt.grid(True)
plt.savefig("clusterss.png")  
plt.show()