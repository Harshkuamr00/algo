from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram , linkage
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import pandas as pd
import numpy as np

# make_blobs is used to generate synthetic data for clustering
# Here we generate 100 samples with 4 centers
x,y=make_blobs(n_samples=100, centers=4,random_state=42)
# compare the clusters by using linkage 
linkage_matrix = linkage(x, method='ward')

# create a DataFrame to hold the data
df=pd.DataFrame(data=np.column_stack((x,y)), columns=['Feature1', 'Feature2', 'Cluster'])
#print(df.head())
df.to_csv('hierarchical_clustering_data.csv', index=False)  # Save the DataFrame to a CSV file

# Plot the data points
plt.figure(figsize=(12,8))
dendrogram(linkage_matrix)# creates a dendrogram to visualize the hierarchical clustering
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.show()

# here we apply euclidean distance and ward linkage to perform agglomerative clustering 
cluster=AgglomerativeClustering(n_clusters=4,affinity='euclidean',linkage='ward')

y_pred=cluster.fit_predict(x)
plt.scatter(x[:,0],x[:,1],c=y_pred,cmap='rainbow')
plt.title('Agglomerative Clustering Results')

plt.show()
