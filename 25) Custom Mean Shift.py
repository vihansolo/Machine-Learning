import numpy as np
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

data, y = make_blobs(n_samples = 15, centers = 3, n_features = 2)

# data = np.array([[1,2],[1.5,1.8],[5,8],[8,8],[1,0.6],[9,11],[8,2],[10,2],[9,3],])

colors = 10 * ["g","r","c","b","k"]

class meanShift :

# Static :-

#     def __init__(self, bandwidth = 4) :
#         self.bandwidth = bandwidth

#     def fit(self, data) :
#         centroids = {}

#         for i in range(len(data)) :
#             centroids[i] = data[i]

#         while True :
#             newCentroids = []

#             for i in centroids :
#                 inBandwidth = []
#                 centroid = centroids[i]

#                 for featureSet in data :
                    
#                     if np.linalg.norm(featureSet - centroid) < self.bandwidth :
#                         inBandwidth.append(featureSet)
                
#                 newCentroid = np.average(inBandwidth, axis = 0)
#                 newCentroids.append(tuple(newCentroid))

#             uniques = sorted(list(set(newCentroids)))
#             prevCentroids = dict(centroids)
#             centroids = {}

#             for i in range(len(uniques)) :
#                 centroids[i] = np.array(uniques[i])

#             optimized = True

#             for i in centroids :
                
#                 if not np.array_equal(centroids[i], prevCentroids[i]) :
#                     optimized = False

#                 if not optimized :
#                     break

#             if optimized :
#                 break

#         self.centroids = centroids

#     def predict(self, data) :
#         pass

# clf = meanShift()
# clf.fit(data)

# centroids = clf.centroids

# plt.scatter(data[:,0], data[:,1], s = 150)

# for c in centroids :
#     plt.scatter(centroids[c][0], centroids[c][1], color = 'k', marker = '*', s = 150)

# plt.show()

# Dynamic :-

    def __init__(self, bandwidth = None, bandwidthNormStep = 100) :
        self.bandwidth = bandwidth
        self.bandwidthNormStep = bandwidthNormStep

    def fit(self, data) :

        if self.bandwidth == None :
            allDataCentroid = np.average(data, axis = 0)
            allDataNorm = np.linalg.norm(allDataCentroid)

            self.bandwidth = allDataNorm / self.bandwidthNormStep

        centroids = {}

        for i in range(len(data)) :
            centroids[i] = data[i]

        weights = [i for i in range(self.bandwidthNormStep)][::-1]

        while True :
            newCentroids = []

            for i in centroids :
                inBandwidth = []
                centroid = centroids[i]

                for featureSet in data :
                    distance = np.linalg.norm(featureSet - centroid)

                    if distance == 0 :
                        distance = 0.00000001

                    weightIndex = int(distance / self.bandwidth)

                    if weightIndex > self.bandwidthNormStep - 1 :
                        weightIndex = self.bandwidthNormStep - 1

                    toAdd = (weights[weightIndex] ** 2) * [featureSet]
                    inBandwidth += toAdd
                
                newCentroid = np.average(inBandwidth, axis = 0)
                newCentroids.append(tuple(newCentroid))

            uniques = sorted(list(set(newCentroids)))

            toPop = []

            for i in uniques :

                for ii in uniques :

                    if i == ii :
                        pass

                    elif np.linalg.norm(np.array(i) - np.array(ii)) <= self.bandwidth :
                        toPop.append(ii)
                        break

            for i in toPop :

                try :
                    uniques.remove(i)

                except :
                    pass

            prevCentroids = dict(centroids)
            centroids = {}

            for i in range(len(uniques)) :
                centroids[i] = np.array(uniques[i])

            optimized = True

            for i in centroids :
                
                if not np.array_equal(centroids[i], prevCentroids[i]) :
                    optimized = False

                if not optimized :
                    break

            if optimized :
                break

        self.centroids = centroids
        self.classifications = {}

        for i in range(len(self.centroids)) :
            self.classifications[i] = []

        for featureSet in data :
            distances = [np.linalg.norm(featureSet - self.centroids[centroid]) for centroid in self.centroids]
            classification = distances.index(min(distances))
            self.classifications[classification].append(featureSet)

    def predict(self, data) :
        distances = [np.linalg.norm(featureSet - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        
        return classification

clf = meanShift()
clf.fit(data)

centroids = clf.centroids

for classification in clf.classifications :
    color = colors[classification]

    for featureSet in clf.classifications[classification] :
        plt.scatter(featureSet[0], featureSet[1], marker = 'x', color = color, s = 150, linewidths = 5)

for c in centroids :
    plt.scatter(centroids[c][0], centroids[c][1], color = 'k', marker = '*', s = 150)

plt.show()
