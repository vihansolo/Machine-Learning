import numpy as np
from numpy import linalg
from scipy.stats import multivariate_normal
import cvxopt
import cvxopt.solvers

def sign(var) :
    if var > 0 :
        return 1

    elif var < 0 :
        return -1
    
    elif var ==0 :
        return 0
    
    else :
        return var

def linearKernel(x1, x2) :
    return np.dot(x1, x2)

def polynomialKernel(x, y, p = 2) :
    return (1 + np.dot(x, y)) ** p

def gaussianKernel(x, y, sigma = 5.0) :
    return np.exp(-linalg.norm(x - y) ** 2 / (2 * (sigma ** 2)))

class SVM(object) :

    def __init__(self, kernel = linearKernel, C = None) :
        self.kernel = kernel
        self.C = C
        if self.C is not None :
            self.C = float(self.C)

    def fit(self, x, y) :
        n_samples, n_features = x.shape

        # Gram Matrix 

        K = np.zeros((n_samples, n_samples))

        for i in range(n_samples) :
            for j in range(n_samples) :
                K[i,j] = self.kernel(x[i], x[j])

        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1, n_samples))
        b = cvxopt.matrix(0.0)

        if self.C is None :
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))

        else :
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        # Solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Langrange Multipliers
        a = np.ravel(solution['x'])

        # Support vectors have non zero langrange multipliers
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = x[sv]
        self.sv_y = y[sv]
        
        print("%d support vectors out of %d points " % (len(self.a), n_samples))

        # Intercept
        self.b = 0

        for n in range(len(self.a)) :
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n], sv])
        self.b /= len(self.a)

        # Weight Vector
        if self.kernel == linearKernel :
            self.w = np.zeros(n_features)

            for n in range(len(self.a)) :
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        
        else :
            self.w = None
    
    def project(self, x) :
        if self.w is not None :
            return np.dot(x, self.w) + self.b

        else :
            y_predict = np.zeros(len(x))

            for i in range(len(x)) :
                s = 0

                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv) :
                    s += a * sv_y * self.kernel(x[i], sv)                
                y_predict[i] = s
            
            return y_predict + self.b

    def predict (self, x) :
        return sign(self.project(x))

if __name__ == "__main__" :
    import pylab as pl

    def genLinSeperableData() :

        # generate training data in the 2d case
        mean1 = np.array([0,2])
        mean2 = np.array([2,0])

        cov = np.array([[0.8,0.6], [0.6,0.8]])

        x1 = multivariate_normal(mean1, cov, 100)
        y1 = np.ones(len(x1))
        x2 = multivariate_normal(mean2, cov, 100)
        y2 = np.ones(len(x2)) * -1

        return x1, y1, x2, y2

    def genNonLinSeperableData() :
        mean1 = [-1,2]
        mean2 = [1,-1]
        mean3 = [4,-4]
        mean4 = [-4,4]

        cov = ([[1.0,0.8], [0.8,1.0]])

        x1 = multivariate_normal(mean1, cov, 50)
        x1 = np.vstack((x1, multivariate_normal(mean3, cov, 50)))
        y1 = np.ones(len(x1))
        x2 = multivariate_normal(mean2, cov, 50)
        x2 = np.vstack((x2, multivariate_normal(mean4, cov, 50)))
        y2 = np.ones(len(x2)) * -1

        return x1, y1, x2, y2

    def genLinSeperableOverlapData() :

        # generate training data in the 2d case
        mean1 = np.array([0,2])
        mean2 = np.array([2,0])

        cov = np.array([[1.5,1.0], [1.0,1.5]])

        x1 = multivariate_normal(mean1, cov, 100)
        y1 = np.ones(len(x1))
        x2 = multivariate_normal(mean2, cov, 100)
        y2 = np.ones(len(x2)) * -1

        return x1, y1, x2, y2

    def splitTrain(x1, x2, y1,y2) :
        x1_train = x1[:90]
        y1_train = y1[:90]
        x2_train = x2[:90]
        y2_train = y2[:90]

        x_train = np.vstack((x1_train, x2_train))
        y_train = np.hstack((y1_train, y2_train))

        return x_train, y_train

    def splitTest(x1, x2, y1,y2) :
        x1_test = x1[90:]
        y1_test = y1[90:]
        x2_test = x2[90:]
        y2_test = y2[90:]

        x_test = np.vstack((x1_test, x2_test))
        y_test = np.hstack((y1_test, y2_test))

        return x_test,y_test

    def plotMargin(x1_train, x2_train, clf) :
        def f(x, w, b, c = 0) :

            # w.x + b = c
            return (-w[0] * x - b + c) / w[1]

        pl.plot(x1_train[:,0], x1_train[:,1], "ro")
        pl.plot(x2_train[:,0], x2_train[:,1], "bo")
        pl.scatter(clf.sv[:,0], clf.sv[:,1], s = 100, c = "g")

        # w.x + b = 0
        a0 = -4
        a1 = f(a0, clf.w, clf.b)
        b0 = 4
        b1 = f(b0, clf.w, clf.b)
        pl.plot([a0,b0], [a1,b1], "k")

        # w.x + b = 1
        a0 = -4
        a1 = f(a0, clf.w, clf.b, 1)
        b0 = 4
        b1 = f(b0, clf.w, clf.b, 1)
        pl.plot([a0,b0], [a1,b1], "k--")

        # w.x + b = -1
        a0 = -4
        a1 = f(a0, clf.w, clf.b, -1)
        b0 = 4
        b1 = f(b0, clf.w, clf.b, -1)
        pl.plot([a0,b0], [a1,b1], "k--")

        pl.axis("tight")
        pl.show()

    def plotContour(x1_train, x2_train, clf) :
        pl.plot(x1_train[:,0], x1_train[:,1], "ro")
        pl.plot(x2_train[:,0], x2_train[:,1], "bo")
        pl.scatter(clf.sv[:,0], clf.sv[:,1], s = 100, c = "g")

        x1, x2 = np.meshgrid(np.linspace(-6,6,50), np.linspace(-6,6,50))
        x = np.array([[x1, x2] for x1, x2 in zip(np.ravel(x1), np.ravel(x2))])
        z = clf.project(x).reshape(x1.shape)

        pl.contour(x1, x2, z, [0.0], colors = 'k', linewidths = 1, origin = 'lower')
        pl.contour(x1, x2, z + 1, [0.0], colors = 'grey', linewidths = 1, origin = 'lower')
        pl.contour(x1, x2, z - 1, [0.0], colors = 'grey', linewidths = 1, origin = 'lower')

        pl.axis("tight")
        pl.show()

    def testLinear() :
        x1, y1, x2, y2 = genLinSeperableData()
        x_train, y_train = splitTrain(x1, y1, x2, y2)
        x_test, y_test = splitTest(x1, y1, x2, y2)

        clf = SVM()
        clf.fit(x_train, y_train)

        y_predict = clf.predict(x_test)
        correct = np.sum(y_predict == y_test)
        print("%d out of %d predictions correct" % (correct, len(y_predict)))

        plotMargin(x_train[y_train == 1], x_train[y_train == -1], clf)

    def testNonLinear() :
        x1, y1, x2, y2 = genNonLinSeperableData()
        x_train, y_train = splitTrain(x1, y1, x2, y2)
        x_test, y_test = splitTest(x1, y1, x2, y2)

        clf = SVM(polynomialKernel)
        clf.fit(x_train, y_train)

        y_predict = clf.predict(x_test)
        correct = np.sum(y_predict == y_test)
        print("%d out of %d predictions correct" % (correct, len(y_predict)))

        plotContour(x_train[y_train == 1], x_train[y_train == -1], clf)

    def testSoft() :
        x1, y1, x2, y2 = genLinSeperableOverlapData()
        x_train, y_train = splitTrain(x1, y1, x2, y2)
        x_test, y_test = splitTest(x1, y1, x2, y2)

        clf = SVM(C = 1000.1)
        clf.fit(x_train, y_train)

        y_predict = clf.predict(x_test)
        correct = np.sum(y_predict == y_test)
        print("%d out of %d predictions correct" % (correct, len(y_predict)))

        plotContour(x_train[y_train == 1], x_train[y_train == -1], clf)        

testLinear()
# testNonLinear()
# testSoft()
