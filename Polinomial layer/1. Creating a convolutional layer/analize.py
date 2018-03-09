import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import exponents
import scipy.spatial.distance

def compute_distance(exponent, kernel_size):
    matrix = np.reshape(exponent,kernel_size)
    indexes = []
    for i in range(kernel_size[1]):
        for j in range(kernel_size[0]):
            if matrix[i][j] != 0:
                indexes.append([i,j])
    return np.sum(scipy.spatial.distance.pdist(indexes, metric="cityblock"))

degree = 2
variables = 25
kernel = [5,5]
exp = exponents.uptodegree(25,2)

data = np.asarray(pd.read_csv("/home/esteve/PycharmProjects/TFG/Polinomial layer/1. Creating a convolutional layer/2par/E0B200.csv", header=None))
#[len(self._exponent),self._filters]

x = []
for i in range(9):
    x.append([])

for i in range(len(exp)):
    if np.sum(exp[i]) == 2:
        x[np.int(compute_distance(exp[i],kernel))].extend([np.abs(j) for j in data[i]])

#for coefficient in np.asarray(data):
#    abs.append(np.mean([x * x for x in coefficient]))




plt.boxplot(x)
plt.savefig('distance.png')


