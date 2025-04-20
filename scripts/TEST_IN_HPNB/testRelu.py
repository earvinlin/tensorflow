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

def naive_vector_dot(x, y) :
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    assert x.shape[0] == y.shape[0]

    z = 0.
    for i in range(x.shape[0]) :
        z += x[i] * y[i]
    return z

# 矩陣向量點積函式
def naive_matrix_vector_dot(x, y) :
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]

    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]) :
        for j in range(x.shape[1]) :
            z[i] += x[i, j] * y[j]
    return z

# 兩個矩陣點積
def naive_matrix_dot(x, y) :
    assert len(x.shape) == 2
    assert len(y.shape) == 2
    assert x.shape[1] == y.shape[0]

    z = np.zeros((x.shape[0], y.shape[1]))
    for i in range(y.shape[0]) :
        for j in range(y.shape[1]) :
            row_x = x[i, :]
            column_y = y[:, j]
            z[i, j] = naive_vector_dot(row_x, column_y)
    return z


