import numpy as np
import pandas as pd

def naive_matrix_vector_dot(x, y) :
    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]) :
        for j in range(x.shape[1]) :
            z[i] += x[i, j] * y[j]
    return z

x = np.array([[1,2,3],[4,5,6]])
y = np.array([2,1,1])

z = naive_matrix_vector_dot(x, y)
print(z)
