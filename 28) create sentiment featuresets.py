import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle
from collections import Counter

lemmatizer = WordNetLemmatizer()
noOfLines = 10000000

print()

def createLexicon(pos, neg) :
    lexicon = []

    for fi in [pos, neg] :
        with open(fi, 'r') as f :
            contents = f.readline()

            for l in contents[:noOfLines] :
                allWords = word_tokenize(l.lower())
                lexicon += list(allWords)
    
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    wordCounts = Counter(lexicon)

    # wordCounts e.g. -
    #       wordCounts = {'the' : 248239, 'and' : 54675}

    l2 = []

    for w in wordCounts :
        if 1000 > wordCounts[w] > 50 :
            l2.append(w)

    print(len(l2))

    return l2

def sampleHandling(sample, lexicon, classification) :
    featureSet = []

    with open(sample, 'r') as f :
        contents = f.readlines()

        for l in contents[:noOfLines] :
            currentWords = word_tokenize(l.lower())
            currentWords = [lemmatizer.lemmatize(i) for i in currentWords]
            
            features = np.zeros(len(lexicon))

            for word in currentWords :
                if word.lower() in lexicon :
                    indexValue = lexicon.index(word.lower())
                    features[indexValue] += 1

            features = list(features)
            featureSet.append([features, classification])

    return featureSet

def createFeatureSetsAndLabels(pos, neg, testSize = 0.1) :
    lexicon = createLexicon(pos, neg)

    features = []
    features += sampleHandling('pos.txt', lexicon, [1, 0])
    features += sampleHandling('neg.txt', lexicon, [0, 1])

    random.shuffle(features)

    features = np.array(features)

    testingSize = int(testSize * len(features))

    trainX = list(features[:,0][:-testingSize])
    trainY = list(features[:,1][:-testingSize])

    testX = list(features[:,0][-testingSize:])
    testY = list(features[:,1][-testingSize:])

    return trainX, trainY, testX, testY

if __name__ == '__main__' :
    trainX, trainY, testX, testY = createFeatureSetsAndLabels('pos.txt', 'neg.txt')

    with open('sentiment_set.pickle', 'wb') as f :
        pickle.dump([trainX, trainY, testX, testY], f)
