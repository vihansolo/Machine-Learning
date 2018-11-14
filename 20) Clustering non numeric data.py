import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
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

print(df.head())
