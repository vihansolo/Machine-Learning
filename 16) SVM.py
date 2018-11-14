import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')

print()

class suportVectorMachine :

    def __init__(self, visualization = True) :

        self.visualization = visualization
        self.colors = {1 : 'r', -1 : 'b'}

        if self.visualization :
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)

    def fit(self, data) :

        self.data = data

        # { ||w|| : [w,b]}
        optDict = {}
        transforms = [[1,1], [-1,1], [-1,-1], [1,-1]]
        allData = []

        for yi in self.data :
            for featureset in self.data[yi] :
                for feature in featureset :
                    allData.append(feature)
        
        self.maxFeatureValue = max(allData)
        self.minFeatureValue = min(allData)
        allData = None

        stepSizes = [self.maxFeatureValue * 0.1,
                      self.maxFeatureValue * 0.01,

                      # point of expense :
                      self.maxFeatureValue * 0.001,]

        # extremely expensive
        bRangeMultiple = 2

        # no small steps with b
        bMultiple = 5
        latestOptimum = self.maxFeatureValue * 10

        for step in stepSizes :
            w = np.array([latestOptimum, latestOptimum])
            
            # convex
            optimized = False

            while not optimized :
                for b in np.arange(-1 * (self.maxFeatureValue * bRangeMultiple), self.maxFeatureValue * bRangeMultiple, step * bMultiple) :
                    for transformation in transforms :
                        w_t = w * transformation
                        foundOption = True

                        # Weakest link in SVM, SMO attempts to fix a bit
                        # yi(xi . w + b) >= 1
                        for i in self.data :
                            for xi in self.data[i] :
                                yi = i
                                if not yi * (np.dot(w_t, xi) + b) >= 1 :
                                    foundOption = False
                                # print(xi, ':', yi * (np.dot(w_t, xi) + b))

                        if foundOption :
                            optDict[np.linalg.norm(w_t)] = [w_t,b]

                if w[0] < 0 :
                    optimized = True
                    print('Optimized a step.')

                else :
                    w = w - step
            
            norms = sorted([n for n in optDict])
            optChoice = optDict[norms[0]]

            self.w = optChoice[0]
            self.b = optChoice[1]
            latestOptimum = optChoice[0][0] + step * 2

    def predict(self, features) :

        # sign (x . w + b)
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)

        if classification != 0 and self.visualization :
            self.ax.scatter(features[0], features[1], s = 100, marker = '*', c = self.colors[classification])

        return classification

    def visualize(self) :

        [[self.ax.scatter(x[0], x[1], s = 100, c = self.colors[i]) for x in dataDict[i]] for i in dataDict]

        # v = x . w + b
        def hyperplane(x, w ,b, v) :
            return(-w[0] * x - b + v) / w[1]

        dataRange = (self.minFeatureValue * 0.9, self.maxFeatureValue * 1.1)
        hyperplaneXMin = dataRange[0]
        hyperplaneXMax = dataRange[1]

        # +ve support vector hyperplane
        psv1 = hyperplane(hyperplaneXMin, self.w, self.b, 1)
        psv2 = hyperplane(hyperplaneXMax, self.w, self.b, 1)
        self.ax.plot([hyperplaneXMin, hyperplaneXMax], [psv1, psv2], 'k')

        # -ve support vector hyperplane
        nsv1 = hyperplane(hyperplaneXMin, self.w, self.b, -1)
        nsv2 = hyperplane(hyperplaneXMax, self.w, self.b, -1)
        self.ax.plot([hyperplaneXMin, hyperplaneXMax], [nsv1, nsv2], 'k')

        # Decision Boundary
        db1 = hyperplane(hyperplaneXMin, self.w, self.b, 0)
        db2 = hyperplane(hyperplaneXMax, self.w, self.b, 0)
        self.ax.plot([hyperplaneXMin, hyperplaneXMax], [db1, db2], 'y--')

        plt.show()

dataDict = {-1 : np.array([[1,7],
                           [2,8],
                           [3,8],]), 
             1 : np.array([[5,1],
                           [6,-1],
                           [7,3],])}

svm = suportVectorMachine()
svm.fit(data = dataDict)

prediction = [[0,10],[1,3],[3,4],[3,5],[5,5],[5,6],[6,-5],[5,8],]

for p in prediction :
    svm.predict(p)

svm.visualize()
