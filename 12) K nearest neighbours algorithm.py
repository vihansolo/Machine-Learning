import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
import warnings

style.use('fivethirtyeight')

dataset = {'k' : [[1,2],[2,3],[3,1]], 'r' : [[6,5],[7,7],[8,6]]}
new_features = [5,7]

def K_nearest_neighbours (data, predict, k = 3) :

    if len(data) >= k :
        warnings.warn('K is set to a value less than total voting groups!!')

    distances = []

    for group in data :
        for features in data[group] :
            # euclidean_distance = np.sqrt(np.sum((np.array(features) - np.array(predict)) ** 2))
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    print(Counter(votes).most_common(1))
    print()
    vote_result = Counter(votes).most_common(1)[0][0]

    return vote_result

result = K_nearest_neighbours(dataset, new_features, k = 3)
print(result)

[[plt.scatter(ii[0], ii[1], s = 100, color = i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_features[0], new_features[1], color = result)
plt.show()
