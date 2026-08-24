import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

iris = pd.read_csv('Iris.csv')
X = iris.drop(['Species','Id'],axis=1)
y = iris['Species'].values

df = pd.DataFrame(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
df['cluster'] = kmeans.fit_predict(X)

plt.figure(figsize=(8, 6))

plt.scatter(df['PetalLengthCm'], 
            df['PetalWidthCm'], 
            c=df['cluster'], 
            cmap='viridis', 
            s=50, alpha=0.8, 
            edgecolor='k')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 2], centers[:, 3], 
            c='red')

plt.title('K-Means Clustering on Iris Dataset (Petal Dimensions)')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.grid()
plt.show()