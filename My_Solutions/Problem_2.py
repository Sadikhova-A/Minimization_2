import numpy as np


# Defining the Hessian matrix of function  f
def Hessian(x, y, z, w):
    return np.array([
        [220.2, 0, 19.8, -400 * w],
        [0, 2 + 720 * y ** 2 + 360 * (y ** 2 - z), -360 * y, 0],
        [19.8, -360 * y, 200.2, 0],
        [-400 * w, 0, 0, 800 * w ** 2 + 400 *(w ** 2 - x) + 2]
    ])


# Defining the grad vector of function f
def grad_f(x, y, z, w):
    df_x = - 200 * (w ** 2 - x) + 20.2 * (x - 1) + 19.8 * (z - 1)
    df_y = 2 * (y - 1) + 360 * y * (y ** 2 - z)
    df_z = -180 * (y ** 2 - z) + 20.2 * (z - 1) + 19.8 * (x - 1)
    df_w = 400 * w * (w ** 2 - x) + 2 * (w - 1)
    return np.array([df_x, df_y, df_z, df_w])


def d_k(x, y, z, w):
    grad = grad_f(x, y, z, w)
    hess = Hessian(x, y, z, w)
    return np.linalg.solve(hess, -grad), grad


# Pure Newton Algorithm
def Pure_Newton(x, y, z, w):
    xk = np.array([x, y, z, w])
    d, gradf = d_k(xk[0], xk[1], xk[2], xk[3])
    num = 0
    while np.linalg.norm(gradf) > 1e-5:
        num += 1
        xk += d
        d, gradf = d_k(xk[0], xk[1], xk[2], xk[3])
    return xk, num


print(Pure_Newton(0.0, 1.0, 2.0, 3.0))
