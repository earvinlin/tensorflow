def naive_relu(x) :
    assert len(x.shape) == 2

    x = x.copy()
    for i in range(x.shape[0]) :
        for j in range(x.shape[1]) :
            x[i, j] = max(x[i, j], 0)
    return x

def naive_add(x, y) :
    assert len(x.shape) == 2
    assert x.shape == y.shape

    x = x.copy()
    for i in range(x.shape[0]) :
        for j in range(x.shape[1]) :
            x[i, j] += y[i, j]
    return x

def naive_add_matrix_and_vector(x, y) :
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]

    x = x.copy()
    for i in range(x.shape[0]) :
        for j in range(x.shape[1]) :
            x[i, j] += y[j]
    return x

import numpy as np

x = np.random.random((4, 3, 2, 3))
print("x= ", x)
y = np.random.random((2, 3))
print("y= ", y)
z = np.maximum(x, y)
print("z= ", z)

