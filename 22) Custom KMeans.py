import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

data = np.array([[1,2],[1.5,1.8],[5,8],[8,8],[1,0.6],[9,11],[1,3],[8,9],[0,3],[5,4],[6,4],])

# plt.scatter(data[:,0], data[:,1], s = 150)
# plt.show()

colors = 10 * ["g","r","c","b","k"]

class KMeans :

    def __init__(self, k = 2, tolerance = 0.001, maxIterations = 300) :
        self.k = k
        self.tolerance = tolerance
        self.maxIterations = maxIterations

    def fit(self, data) :

        self.centroids = {}

        for i in range(self.k) :
            self.centroids[i] = data[i]

        for i in range(self.maxIterations) :
            self.classifications = {}

            for i in range(self.k) :
                self.classifications[i] = []
            
            for featureSet in data :
                distances = [np.linalg.norm(featureSet - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureSet)

            prevCentroids = dict(self.centroids)

            for classification in self.classifications :
                self.centroids[classification] = np.average(self.classifications[classification], axis = 0)

            optimized = True

            for c in self.centroids :
                originalCentroid = prevCentroids[c]
                currentCentroid = self.centroids[c]

                if np.sum((currentCentroid - originalCentroid) / originalCentroid * 100.0) > self.tolerance :

                    print(np.sum((currentCentroid - originalCentroid) / originalCentroid * 100.0))
                    optimized = False

            if optimized :
                break

    def predict(self, data) :
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))

        return classification

clf = KMeans()
clf.fit(data)

for centroid in clf.centroids :
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1], marker = "o", color = "k", s = 150, linewidths = 5)

for classification in clf.classifications :
    color = colors[classification]

    for featureSet in clf.classifications[classification] :
        plt.scatter(featureSet[0], featureSet[1], marker = "x", color = color, s = 150, linewidths = 5)

# randomDatas = np.array([[1,3],[8,9],[0,3],[5,4],[6,4],])

# for randomData in randomDatas :
#     classification = clf.predict(randomData)
#     plt.scatter(randomData[0], randomData[1], marker = "*",  color = colors[classification], s = 150, linewidths = 5)

plt.show()
