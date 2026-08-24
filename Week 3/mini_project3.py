import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

iris = pd.read_csv('Iris.csv')
X = iris.drop(['Species', 'Id'], axis=1)
y_true = iris['Species'].values
target_names = np.unique(y_true).tolist()

kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
y_pred = kmeans.fit_predict(X)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

scatter1 = axes[0].scatter(X['PetalLengthCm'], X['PetalWidthCm'], c=y_pred, cmap='viridis', s=50, alpha=0.8, edgecolor='k')
centers = kmeans.cluster_centers_

axes[0].scatter(centers[:, 2], centers[:, 3], c='red', s=200, marker='X', label='Centroids')
axes[0].set_title('K-Means Clustering (Predicted)')
axes[0].set_xlabel('Petal Length (cm)')
axes[0].set_ylabel('Petal Width (cm)')
axes[0].legend()
axes[0].grid()

le = LabelEncoder()
y_true_encoded = le.fit_transform(y_true)

scatter2 = axes[1].scatter(X['PetalLengthCm'], X['PetalWidthCm'], c=y_true_encoded, cmap='Set1', s=50, alpha=0.8, edgecolor='k')
axes[1].set_title('True Iris Species Labels')
axes[1].set_xlabel('Petal Length (cm)')
axes[1].set_ylabel('Petal Width (cm)')

legend_elements = [Line2D([0], [0], marker='o', color='w', label=name, markerfacecolor=color, markersize=8) 
                   for name, color in zip(target_names, ['red', 'blue', 'green'])]
axes[1].legend(handles=legend_elements)
axes[1].grid()

plt.tight_layout()
plt.show()

print("Confusion Matrix :")
print(confusion_matrix(y_true_encoded, y_pred))