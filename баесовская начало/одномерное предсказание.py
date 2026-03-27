import math
from math import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import random

x=np.linspace(0, 2*math.pi, 100)

def func(x):
    return sin(x)

def plot(X, Y):
    plt.plot(X, Y)
    plt.show()

def rbf_kernel(x_predict, x_init, sigma=1, l=1): # матрица x1*x2
    result=[]
    if isinstance(x_predict, list):
        pass
    else:
        x_predict=[x_predict]
    if isinstance(x_init, list):
        pass
    else:
        x_init=[x_init]
    for i in x_predict:
        row=[]
        for j in x_init:
            row.append(sigma ** 2 * math.exp(-(i-j)**2 / (2 * l ** 2)))
        result.append(row)
    return result

def baesian(data, X_new):
    y_train, x_train = data
    covXx = rbf_kernel(X_new, x_train)
    covxx = rbf_kernel(x_train, x_train)
    mu_y = np.mean(y_train)
    print(np.full(len(y_train), float(mu_y)))
    print(y_train)
    y_centered = y_train -  np.full(len(y_train), float(mu_y))
    qw = np.linalg.solve(covxx, y_centered)
    mu = mu_y + covXx @ qw
    return mu[0]



def distributions(data, X_new):
    y_train, x_train = data
    covXx = rbf_kernel(X_new, x_train)
    covxx = rbf_kernel(x_train, x_train)
    covxX=rbf_kernel(x_train,X_new)
    covXX=rbf_kernel(X_new, X_new)
    qw = np.linalg.solve(covxx, covxX)
    sigma=covXX-covXx@qw
    return sigma

def UCB(mean, sigma, *,b=1):
    return mean-b*sigma

# берем точку и вычисляем реальное значение в ней
data=[[],[]]
first_x=random.choice(x)
first_y=func(first_x)
data[0].append(first_y)
data[1].append(first_x)

# цикл
for i in range (10):
    temporary_sigma=[]
    temporary_data=[[],[]]
    temporary_UCB=[]
    for temporary_x in x:
        temporary_y=baesian(data, temporary_x)
        temporary_data[0].append(temporary_y)
        temporary_data[1].append(temporary_x)
        sigma=distributions(data, temporary_x)
        temporary_sigma.append(sigma)
        temporary_UCB.append(UCB(temporary_y, temporary_sigma))

    plt.plot(x, temporary_data[0])
    plt.plot(x, temporary_sigma)
    next_x=np.argmin(temporary_UCB)
    next_y=func(next_x)
    data[0].append(next_y)
    data[1].append(next_x)




# прогоняем диапазон значений и ищем точку с minarg
# строим график
# вычисляем значение в точке и добавляем ее в общие данные

