"""Optimizing the function's parameters to fit the datapoints"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# Import .csv file
data = pd.read_csv("csv-data.csv")


# Regression Function
def expo(t, c0, c1, c2, c3):
    return t*t+c0+c1+c2+c3


# Guessing the Parameters
g = [5, 5, 0, 0]


# Plotting the curve (not optimized)
n = len(data['Input'])
y = np.empty(n)
for i in range(n):
    y[i] = expo(data['Input'][i], g[0], g[1], g[2], g[3])
# plt.plot(data['Input'], data['Output'])
# plt.plot(data['Input'], y, 'ro')
# plt.show()


# Optimizing the Parameters
inp = data['Input'].values
out = data['Output'].values
c, cov = curve_fit(expo, inp, out, g, maxfev=100000)
print(cov)
print(c)


# Plotting the curve (optimized)
for i in range(n):
    y[i] = expo(data['Input'][i], c[0], c[1], c[2], c[3])
plt.plot(data['Input'], data['Output'])
plt.plot(data['Input'], y, 'ro')
plt.show()
