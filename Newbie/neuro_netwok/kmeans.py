import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

# msStart = int(time.clock()*1000)





def average(clusters):
    c_array = []

    for c in clusters:
        c_array.append(np.mean(np.array(c)[:, 0]))
    return np.array(c_array)


def createClust(data, k):
    print('kmeans')
    centroids = np.array([])
    clusters = [[] for i in range(k)]
    for cz_i in range(k):
        cz = data[np.random.randint(len(data))][0]
        while cz in centroids:
            cz = data[np.random.randint(len(data))][0]
        centroids = np.append(centroids, cz)

    iter = 1
    clusters_r = []
    #c_data = np.array(data)[:, 0].reshape((len(data), 1))

    while True:

        #compare_array = np.argmin(np.sqrt(np.abs(c_data-centroids)), axis=1)
        for it, i in enumerate(data):
            evk_array = np.sqrt((i[0] - centroids) ** 2)
            clusters[np.argmin(evk_array)].append(i)


        oldCentroids = centroids
        centroids = average(np.array(clusters))

        if not np.array_equal(centroids, oldCentroids):
            clusters = [[] for i in range(k)]
        else:
            #clusters_r = clusters
            #centroids_r = centroids
            print('kmeans %i iteration' % iter)
            break

        iter += 1


    return_cent = np.sort(centroids)[::-1]
    for i in range(len(centroids)):
        remove_index = np.argmax(centroids)
        clusters_r.append(np.array(clusters[remove_index])[:, 1:])
        #print(centroids[remove_index], ' cz add and remote')
        centroids[remove_index] = 0


    return clusters_r, return_cent

# print('%s ms' % (int(time.clock()*1000) - msStart))



# matplt.visual(clusters, centroids)
