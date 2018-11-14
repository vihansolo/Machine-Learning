import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

print()
# x_array = np.array([1, 2, 3, 4, 5, 6], dtype = np.float64)
# y_array = np.array([5, 4, 6, 5, 6, 7], dtype = np.float64)

def create_dataset(number_of_data_points, variance, step = 2, correlation = False) :

    value = 1
    y_array = []

    for i in range(number_of_data_points) :

        y = value + random.randrange(-variance, variance)
        y_array.append(y)

        if correlation and correlation == 'pos':
            value += step
        
        elif correlation and correlation == 'neg':
            value -= step

    x_array = [i for i in range(len(y_array))]

    return np.array(x_array, dtype = np.float64), np.array(y_array, dtype = np.float64)

def best_fit_slope_intercept (x, y) :

    m = (((np.mean(x_array) * np.mean(y_array)) - np.mean(x_array * y_array)) / ((np.mean(x_array) ** 2) - np.mean(x_array ** 2)))
    b = np.mean(y_array) - m * np.mean(x_array)

    return m, b

def squared_error(y_original, y_line) :

    return sum((y_line - y_original) ** 2)

def coefficient_of_determination(y_original, y_line) :

    y_mean_line = [np.mean(y_array) for y in y_original]
    squared_error_regr = squared_error(y_original, y_line)
    squared_error_y_mean = squared_error(y_original, y_mean_line)

    return 1 - (squared_error_regr / squared_error_y_mean)

x_array, y_array = create_dataset(40, 10, 2, correlation = 'pos')

m,b = best_fit_slope_intercept(x_array, y_array)

regression_line = [(m * x) + b for x in x_array]

predict_x = 8
predict_y = (m * predict_x) + b

# for x in xs:
#     regression_line.append((m * x) + b)

r_squared = coefficient_of_determination(y_array, regression_line)
print(r_squared)

plt.scatter(x_array, y_array)
# plt.scatter(predict_x, predict_y)
plt.plot(x_array, regression_line)
plt.show()
