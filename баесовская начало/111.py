import math
from math import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import random

x=np.linspace(0, 2*math.pi, 100)
e=10**(-6)
def func(x):
    return sin(x)

def plot(X, Y):
    plt.plot(X, Y)
    plt.show()

def rbf_kernel(x_predict, x_init, sigma=1, l=1): # матрица x1*x2
    x_predict=np.atleast_1d(x_predict)
    x_init=np.atleast_1d(x_init)
    value = sigma ** 2 * np.exp(-(x_predict[:,None] - x_init[:,None]) ** 2 / (2 * l ** 2))
    print("размер", len(value), len(value[0]))
    return value

def inv(a, b):
    print("обратная")
    print(a)
    print(b)
    print("det", np.linalg.det(a))
    if np.linalg.det(a) < e:
        for index1, i in enumerate(a):
            for index2, j in enumerate(i):
                if index1 == index2:
                    a[index1][index2] += e * 10
                    print("ошибка")
    print(a)
    print("det change", np.linalg.det(a))
    qw = np.linalg.solve(a, b)
    return qw

def baesian(data, X_new):
    y_train, x_train = data
    covXx = rbf_kernel(X_new, x_train)
    covxx = rbf_kernel(x_train, x_train)
    mu_y = np.mean(y_train)
    y_centered = y_train -  np.full(len(y_train), float(mu_y))
    print(y_centered)
    print("cov",covxx)
    qw=inv(covxx, y_centered)
    mu_y=np.atleast_1d(mu_y)
    mu = mu_y + covXx @ qw
    print(mu)
    return mu



def distributions(data, X_new):
    y_train, x_train = data
    covXx = rbf_kernel(X_new, x_train)
    covxx = rbf_kernel(x_train, x_train)
    covxX=rbf_kernel(x_train,X_new)
    covXX=rbf_kernel(X_new, X_new)
    qw = inv(covxx, covxX)
    sigma=covXX-covXx@qw
    print(type(sigma))
    #print((np.diag(sigma)))
    return np.sqrt(np.diag(sigma))

def UCB(mean, sigma, *,b=1):
    return mean-b*sigma

# берем точку и вычисляем реальное значение в ней
data=[[],[]]
first_x=random.choice(x)
first_y=func(first_x)
data[0].append(first_y)
data[1].append(first_x)
delta=float("inf")

# цикл
while delta>e:
    temporary_data=[]
    temporary_UCB=[]
    temporary_y=baesian(data, x)
    print("y", temporary_y)
    temporary_data.append(temporary_y)
    temporary_data.append(x)
    print("11",temporary_data)
    sigma=distributions(data, x)
    temporary_UCB=UCB(temporary_y, sigma)
    print("sigma",sigma)

    plt.plot(x, temporary_data[0])
    plt.plot(x, sigma, label="sigma")
    plt.legend()
    plt.show()
    next_x=x[np.argmin(temporary_UCB)]
    next_y=func(next_x)
    data[0].append(next_y)
    data[1].append(next_x)

    print("x:",next_x)

    delta=abs((data[0][-1]-data[0][-2])/max(data[0][-2],e))



# прогоняем диапазон значений и ищем точку с minarg
# строим график
# вычисляем значение в точке и добавляем ее в общие данные

