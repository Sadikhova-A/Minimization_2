import numpy as np
import matplotlib.pyplot as plt

# Made an effort to make the code neat, pretty and match the notation yay
# Defining anchor points
a_1 = np.array([(1, 1), (1, 3), (2, 5), (3, 1)])
a1_x = [1, 1, 2, 3]
a1_y = [1, 3, 5, 1]

# Defining weights
w_1 = np.array([1.5, 1, 1, 1])  # Same as code for part (a) except weight changed

# Initial guess
x_g = np.array([0.0, 0.0])


# Function of weighted distances
def h(x, w=w_1, a=a_1):  # Inputs: a; 4d list containing 2d tuples , x_0; 2d array, w; 4d
    F = 0
    for i in range(len(a_1)):
        F += w[i] * np.linalg.norm(x - a[i])
    return F


# Gradient function of f
def grad_h(x, w=w_1, a=a_1):
    summa = 0
    grad = np.array([0.0, 0.0])
    for i in range(len(a)):
        norm = np.linalg.norm(x - a[i])
        summa += w[i] / norm
        grad += w[i] * (x - a[i]) / norm
    return grad, summa


# Weiszfeld Method
def Weiszfeld(x_k=x_g, w=w_1, a=a_1):
    grad_k, summa_k = grad_h(x_k, w, a)
    num = 0
    while np.linalg.norm(grad_k) > 1e-5:
        grad_k, summa_k = grad_h(x_k, w, a)
        x_k += (-1 / summa_k) * grad_k
        num += 1
    return x_k, num


x_k = Weiszfeld()[0]
print(Weiszfeld())

# Visual check to see it if makes sense

# figure and axes
fig = plt.figure(figsize=(7, 11))
ax = fig.add_subplot()

# Locations of the Factories
ax.scatter(a1_x, a1_y, s=100, marker='s', color='blue', label='Factories')
# Location of the Assembly Line
ax.scatter(x_k[0], x_k[1], marker='x', color='red', s=100, label=(f'{x_k[0]:.7g}', f'{x_k[1]:.7g}'))


plt.axis('equal')
plt.grid(True)
plt.legend(loc='upper right')
plt.title('Town')

plt.show()
