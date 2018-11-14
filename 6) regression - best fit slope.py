import numpy as np
import matplotlib.pyplot as plt

print()
x_array = np.array([1, 2, 3, 4, 5, 6], dtype = np.float64)
y_array = np.array([5, 4, 6, 5, 6, 7], dtype = np.float64)

def best_fit_slope (x, y) :

    m = (((np.mean(x_array) * np.mean(y_array)) - np.mean(x_array * y_array)) / ((np.mean(x_array) ** 2) - np.mean(x_array ** 2)))
    return m

m = best_fit_slope(x_array, y_array)

print(m)
