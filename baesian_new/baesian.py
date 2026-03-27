import math
from math import sin, cos
import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(0, 2*math.pi, 100)
z=np.linspace(0, 2*math.pi, 100)
print(x)
date=[]
X,Z=np.meshgrid(x,z)
Y=np.sin(X)+np.cos(Z)
y=[]


for x_i, z_i in zip(x, z):
    date.append([sin(x_i)+1, x_i])
    y.append(x_i**2)

print(y)
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, Z, Y)
fig.colorbar(surf)
plt.show()
"""print(*date, sep="\n")
covariation=np.cov(date, rowvar=False)
vaiation=np.var(x, ddof=0)

print(covariation)
print(vaiation)
meany=np.mean(y)
meanx=np.mean(x)

def optimization(x):
    y= meany+covariation[0][1]/vaiation*(x-meanx)
    return y



Y1=[]

for i in X1:
    Y1.append(optimization(i))


print(Y1)"""

"""k=(x[0:None])
print(k)
plt.plot(X1, Y1)
plt.plot(x, y)
plt.show()"""

def rbf_kernel(x1, x2, sigma=1, l=1):
    result=[]
    for i1 in x1:
        row=[]
        for i2 in x2:
            row.append(sigma ** 2 * math.exp(-(i1-i2)**2 / (2 * l ** 2)))
        result.append(row)
    return result

"""print(
*rbf_kernel(x, y), sep="\n"
)"""

X1=np.linspace(0, 4*math.pi, 100)

def optimization_1(x, y_train, X):
    covXx = rbf_kernel(X, x)
    print()
    print(covXx)
    covxx = rbf_kernel(x, x)
    print(covxx)
    """Meanx = np.mean(X)"""
    mu_y = np.mean(y_train)
    print("mu_y",mu_y)
    y_centered = y_train - mu_y
    print("y_centered", y_centered)
    qw = np.linalg.solve(covxx, y_centered)

    print(covXx @ np.linalg.inv(covxx) )

    mu = mu_y+covXx @ qw

    print("mu",mu)
    return mu[0]

Y1=[]

for i in X1:
    Y1.append(optimization_1(x, y, [i]))
    #print(optimization_1(x, y, [i]))


print("y1", Y1)
plt.plot(X1, Y1)
#plt.plot(x, y)
plt.show()
