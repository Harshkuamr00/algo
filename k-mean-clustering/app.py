import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

# load the dataset
x,y=make_blobs(n_samples=300, centers=4, random_state=42)


# create a DataFrame
df=pd.DataFrame(x, columns=['x1', 'x2'])
df.to_csv('kmeans_dataset.csv', index=False)
# create a KMeans model
kmeans=KMeans(n_clusters=4,random_state=42)
# fit the model
kmeans.fit(df)
# predict the clusters
y_kmeans=kmeans.predict(df)
# add the cluster labels to the DataFrame
df['cluster']=y_kmeans

cluster_centers=kmeans.cluster_centers_
labels=kmeans.labels_
# plot the clusters
plt.scatter(df['x1'],df['x2'],c=labels,cmap='viridis',edgecolors='k',s=50)
plt.scatter(cluster_centers[:,0], cluster_centers[:,1], c='red', s=200, alpha=0.75, marker='X',label='cluster centers')
plt.title('K-Means Clustering')
plt.xlabel('feature 1(x1)')
plt.ylabel('feature 2(x2)')
plt.figure(figsize=(10, 6))
plt.savefig('kmeans_clustering.png')

plt.legend()
plt.show()
