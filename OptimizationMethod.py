from abc import ABC, abstractmethod
import random
from re import search
import math
import matplotlib.pyplot as plt

import numpy as np
from collections import defaultdict
from typing import Dict

from simulation_result import SimulationResult


class OptimizationMethod(ABC):
    def __init__(self, iterations):
        self.iterations = iterations


    @abstractmethod
    def optimize(self, context, progress_queue):
        """основная задача методов оптимизации заключается в минимизации целевого параметра. В данный момент между искомыми параметрами не выбирается лучший, для каждой итерации,
        а шаг за шагом применяется каждый параметр к расчетной модели, отвечающий цели. """
        pass


class BestProbe(OptimizationMethod):
    def optimize(self, context, progress_queue):
        sim_result = SimulationResult()
        range_of_values = context.range_params.creating_a_range(None)
        new_params = {}
        phases = 3
        samples_in_phase = self.iterations // phases
        iteration = 0
        for phase in range(phases):
            for trial in range(samples_in_phase):
                for key, value in range_of_values.items():
                    random_value = random.choice(value)
                    new_params[key] = random_value
                context.runner.calculation(context.script_processor.build({**new_params}))
                sim_result.save_data(base_dir=context.base_dir)
                context.objective.evaluate(sim_result, context, {**new_params})
                print("параметры", context.best_params)
                print(iteration, "итерации")

                iteration += 1
                progress = int((iteration + 1) / self.iterations * 100)
                progress_queue.put(("progress", progress))


            if phase < phases - 1 and context.best_params is not None:
                range_of_values = context.range_params.creating_a_range(context.best_params)



        if context.best_params is not None:
            context.runner.calculation(context.script_processor.build(context.best_params))
        else: print("the parameters are not optimized")
        progress_queue.put(("finished", None))

class GradientDescent(OptimizationMethod):

    epsilon = 1e-6

    def __init__(self, iterations, *,steps= 0.01, l_r=0.04, b1=0.05, b2=0.80):
        super().__init__(iterations)
        self.step_size = steps
        self.lr = l_r
        self.b1=b1
        self.b2=b2
        print(steps, l_r, b1)

    # проблемы: +1) динамически уменьшать шаг (теперь не изменяем значение если изменился знак градиента мб не лучшее решение),
    # + 2) неравномерное уменьшение, скачки 3) записываем и дельты и результирующие параметры
    # 3) возникает ситуация в которой может не подойти ни дельта + ни -, тут забавный момент поскольку скорее это зависит от правильной модели
    # +4) неадекватный l_r, нужно как-то подбирать его в зависимости от параметра и диапазона, при смене знака градиент почти всегда улетает за 1.5, костыли и много if
    # 5) не останавливается при достижении необходимого количества итераций
    # 6) в идеале не брать значения на концах

    # рекомендации, при сильном изменении параметра уменьшать шаг
    def optimize(self, context, progress_queue):
        params=[]
        sim_result = SimulationResult()

        range_of_values = context.range_params.creating_a_range(None)

        new_params = {}
        gradient_dict = {}
        dict_step = {}
        dict_lr = {}

        v = {}
        G = {}

        for key, value in range_of_values.items():
            new_params[key] = np.mean(value)

            gradient_dict[key] = []

            dict_step[key] = self.step_size
            dict_lr[key] = self.lr

            v[key] = 0
            G[key] = 0

        iteration = 0

        while iteration < self.iterations:

            iteration += 1
            print("iteration", iteration)

            # 1 базовый расчет

            context.runner.calculation(
                context.script_processor.build(new_params)
            )

            sim_result.save_data(base_dir=context.base_dir)

            penalty_base = context.objective.evaluate(
                sim_result, context, new_params
            )

            gradients = {}

            # 2 вычисляем все градиенты

            for key, value in new_params.items():

                step = value * dict_step[key]

                max_value = value + step

                if max_value > max(range_of_values[key]):
                    max_value = max(range_of_values[key])

                params_plus = {**new_params, **{key: max_value}}

                context.runner.calculation(
                    context.script_processor.build(params_plus)
                )

                sim_result.save_data(base_dir=context.base_dir)

                penalty_plus = context.objective.evaluate(
                    sim_result, context, params_plus
                )

                gradient = (penalty_plus - penalty_base) / step
                gradient=max(-50, min(50, gradient))
                gradients[key] = gradient

                gradient_dict[key].append(gradient)

                print(
                    key,
                    "grad:", gradient,
                    "penalty+", penalty_plus,
                    "base", penalty_base
                )

            # 3 обновляем параметры

            temporal_params = {}

            for key, value in new_params.items():

                gradient = gradients[key]

                if len(gradient_dict[key])>1 and np.sign(gradient_dict[key][-1]) != np.sign(gradient_dict[key][-2]):
                    dict_lr[key] = dict_lr[key] / 1.5

                v[key] = self.b1 * v[key] + (1 - self.b1) * gradient
                G[key] = self.b2 * G[key] + (1 - self.b2) * gradient ** 2

                value_norm = (
                                     value - min(range_of_values[key])
                             ) / (
                                     max(range_of_values[key]) - min(range_of_values[key])
                             )

                new_value_norm = value_norm - dict_lr[key] * v[key] / (G[key] + self.epsilon) ** 0.5

                new_value = new_value_norm * (
                        max(range_of_values[key]) - min(range_of_values[key])
                ) + min(range_of_values[key])

                new_value = max(
                    value - 0.2 * value,
                    min(value + 0.2 * value, new_value)
                )
                params.append(new_value)
                print("величина", params)
                temporal_params[key] = new_value
            print("параметры", temporal_params)
            # 4 обновляем точку

            new_params = temporal_params
            progress = int((iteration) / self.iterations * 100)
            progress_queue.put(("progress", progress))

        if context.best_params is not None:
            context.runner.calculation(context.script_processor.build(context.best_params))
        else: print("the parameters are not optimized")
        progress_queue.put(("finished", None))


e=1e-3
class Bayesian_optimization():
    def __init__(self, iterations=10):
        self.iterations = iterations
        self.delta=float("inf")

    def func(self, x, context,sim_result):
        context.runner.calculation(
            context.script_processor.build({"r":x})
        )

        sim_result.save_data(base_dir=context.base_dir)

        penalty= context.objective.evaluate(
            sim_result, context, {"r":x}
        )
        print("123",penalty)
        return penalty

    def UCB(self,mean, sigma, *, b=3):
        return mean - b * sigma

    def plot(self,X, Y):
        plt.plot(X, Y)
        plt.show()

    def rbf_kernel(self,x_predict, x_init, sigma=1, l=0.3):  # матрица x1*x2

        x_predict = np.atleast_1d(x_predict)
        x_init = np.atleast_1d(x_init)
        value = sigma ** 2 * np.exp(-(x_predict[:, None] - x_init[None, :]) ** 2 / (2 * l ** 2))
        return value

    def inv(self,a, b):
        if np.linalg.det(a) < 1e-6:
            for index1, i in enumerate(a):
                for index2, j in enumerate(i):
                    if index1 == index2:
                        a[index1][index2] += e * 10
        qw = np.linalg.solve(a, b)
        return qw

    def baesian(self, data, X_new):
        y_train, x_train = data
        covXx = self.rbf_kernel(X_new, x_train)
        covxx = self.rbf_kernel(x_train, x_train)
        mu_y = np.mean(y_train)
        y_centered = y_train - np.full(len(y_train), float(mu_y))
        qw = self.inv(covxx, y_centered)
        mu_y = np.atleast_1d(mu_y)
        mu = mu_y + covXx @ qw
        return mu

    def distributions(self,data, X_new):
        y_train, x_train = data
        covXx = self.rbf_kernel(X_new, x_train)
        covxx = self.rbf_kernel(x_train, x_train)
        covxX = self.rbf_kernel(x_train, X_new)
        covXX = self.rbf_kernel(X_new, X_new)
        qw = self.inv(covxx, covxX)
        sigma = covXX - covXx @ qw
        # print((np.diag(sigma)))
        return np.sqrt(np.abs(np.diag(sigma)))

    def optimize(self, context, progress_queue):
        sim_result = SimulationResult()
        range_of_values = context.range_params.creating_a_range(None)
        new_params = {}

        x=range_of_values.get("r")
        data = [[], []]
        first_x = random.choice(x)
        first_y = self.func(first_x, context,sim_result)
        data[0].append(first_y)
        data[1].append(first_x)
        for i in range(self.iterations):
            temporary_data = []
            temporary_y = self.baesian(data, x)
            temporary_data.append(temporary_y)
            temporary_data.append(x)
            sigma = self.distributions(data, x)
            temporary_UCB = self.UCB(temporary_y, sigma)

            plt.plot(data[1], data[0], label="real")
            plt.plot(x, temporary_data[0])
            plt.plot(x, sigma, label="sigma")
            next_x = x[np.argmin(temporary_UCB)]
            next_y = self.func(next_x,  context,sim_result)
            data[0].append(next_y)
            data[1].append(next_x)

            print("x:", next_x, "y:", next_y)
            plt.legend()
            plt.show()
            self.delta = abs((data[0][-1] - data[0][-2]) / max(data[0][-2], e))

        if context.best_params is not None:
            context.runner.calculation(context.script_processor.build(context.best_params))
        else: print("the parameters are not optimized")
        progress_queue.put(("finished", None))



