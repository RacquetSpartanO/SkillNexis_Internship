import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

iris = pd.read_csv('Iris.csv')
X = iris.drop(['Species','Id'],axis=1)
y = iris['Species'].values

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
print(f"Total Variance Retained: {sum(pca.explained_variance_ratio_)*100:.2f}%")

kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
cluster_labels = kmeans.fit_predict(X_pca)

plt.figure(figsize=(8, 6))

plt.scatter(X_pca[:, 0], X_pca[:, 1], 
            c=cluster_labels, cmap='viridis', s=50, alpha=0.8, edgecolor='k')

centers_pca = kmeans.cluster_centers_
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red')

plt.title('K-Means Clustering on PCA-Reduced Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid()
plt.show()