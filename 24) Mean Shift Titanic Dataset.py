import numpy as np
from sklearn.cluster import MeanShift
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

df = pd.read_excel('titanic.xls')
originalDf = pd.DataFrame.copy(df)

df.drop(['body', 'name'], 1, inplace = True)
df.apply(pd.to_numeric, errors = 'ignore')
df.fillna(0, inplace = True)

def handleNonNumericData(df) :
    columns = df.columns.values

    for column in columns :
        textDigitVals = {}

        def convertToInt(val) :
            return textDigitVals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64 :
            columnContents = df[column].values.tolist()
            uniqueElements = set(columnContents)

            x = 0

            for unique in uniqueElements :
                if unique not in textDigitVals :
                    textDigitVals[unique] = x
                    x += 1

            df[column] = list(map(convertToInt, df[column]))

    return df

df = handleNonNumericData(df)

df.drop(['boat', 'sex'], 1, inplace = True)

x = np.array(df.drop(['survived'], 1).astype(float))
x = preprocessing.scale(x)
y = np.array(df['survived'])

clf = MeanShift()
clf.fit(x)

labels = clf.labels_
clusterCenters = clf.cluster_centers_

originalDf['cluster_group'] = np.nan

for i in range(len(x)) :
    originalDf['cluster_group'].iloc[i] = labels[i]

survivalRates = {}

for i in range(len(clusterCenters)) :
    tempDf = originalDf[(originalDf['cluster_group'] == float(i))]
    survivalCluster = tempDf[(tempDf['survived'] == 1)]
    survivalRate = len(survivalCluster) / len(tempDf)
    survivalRates[i] = survivalRate

print(survivalRates)
