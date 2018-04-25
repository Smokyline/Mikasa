import numpy as np
from scipy.cluster.vq import kmeans,vq, kmeans2

def create_clust(data,k):
    # computing K-Means with K = 2 (2 clusters)
    clusters=[]
    centroids, _ = kmeans(data.T[0], k)
    centroids = np.sort(centroids)
    centroids = centroids[::-1]
    # assign each sample to a cluster
    idx, _ = vq(data.T[0], centroids)
    for i in range(k):
        clusters.append(data[idx==i].T[1:].T)
        #clusters.append(data[idx==i])
    return clusters, centroids

