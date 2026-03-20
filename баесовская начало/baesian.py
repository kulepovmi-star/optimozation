import math
from math import sin
import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(0, 2*math.pi, 50).reshape(-1, 1)
print(x)
date=[]
y=[]
for i in x:
    date.append([sin(i), i])
    y.append(sin(i))
print(y)
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

X1=np.linspace(0, 2*math.pi, 100).reshape(-1, 1)

def optimization_1(x, y, X):
    covXx = rbf_kernel(X, x)[0]
    print()
    print(covXx)
    covxx = rbf_kernel(x, x)
    print(covxx)
    Meanx = np.mean(X)
    Meany =np.mean(y)
    qw=[i-Meany for i in y]
    print("среднее",Meany)

    print(covXx @ np.linalg.inv(covxx) )

    y = covXx @ np.linalg.inv(covxx) @ qw


    return y

Y1=[]

for i in X1:
    Y1.append(optimization_1(x, y, [i]))
    #print(optimization_1(x, y, [i]))


print("y1", Y1)
plt.plot(X1, Y1)
plt.plot(x, y)
plt.show()
