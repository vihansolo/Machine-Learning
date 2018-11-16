import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

df = pd.read_excel('titanic.xls')
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

clf = KMeans(n_clusters = 2)
clf.fit(x)

correct = 0

for i in range(len(x)) :
    predict_me = np.array(x[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)

    if prediction[0] == y[i] :
        correct += 1

print()
print(correct / len(x))
