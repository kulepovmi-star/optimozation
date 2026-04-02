from abc import ABC, abstractmethod
import random
from re import search
import math
import matplotlib.pyplot as plt
from itertools import product
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



class Bayesian_optimization():
    def __init__(self, iterations=10):
        self.iterations = iterations
        self.delta=float("inf")
        self.L=None

    def vector_to_params(self, x_vec, param_names):
        return dict(zip(param_names, x_vec))

    def func(self, x_vec, param_names, context,sim_result):
        """x, y = x_vec

                # Branin function
                a = 1.0
                b = 5.1 / (4 * np.pi ** 2)
                c = 5 / np.pi
                r = 6
                s = 10
                t = 1 / (8 * np.pi)

                value = a * (y - b * x ** 2 + c * x - r) ** 2 + s * (1 - t) * np.cos(x) + s

                print("penalty:", value)
                return value"""
        params = self.vector_to_params(x_vec, param_names)

        context.runner.calculation(
            context.script_processor.build(params)
        )

        sim_result.save_data(base_dir=context.base_dir)

        penalty = context.objective.evaluate(
            sim_result, context, params
        )
        print("penalty:", penalty)
        return penalty




    def LCB(self,mean, sigma, *, b=1.5):

        return mean - b * sigma


    def rbf_kernel(self,x_predict, x_init, *, sigma=1, l=1):  # матрица x1*x2

        X1 = np.atleast_2d(x_predict)
        X2 = np.atleast_2d(x_init)

        diff = X1[:, None, :] - X2[None, :, :]
        sqdist = np.sum(diff ** 2, axis=2)

        return sigma ** 2 * np.exp(-sqdist / (2 * l ** 2))


    def baesian(self, data, X_new, sigma_kernel):
        y_train, x_train = data
        covXx = self.rbf_kernel(X_new, x_train, sigma=sigma_kernel, l=1)
        covxx = self.rbf_kernel(x_train, x_train, sigma=sigma_kernel, l=1)
        noise = 1e-6
        K = covxx + noise * np.eye(len(covxx))
        self.L = np.linalg.cholesky(K)
        mu_y = np.mean(y_train)
        y_std=np.std(y_train)
        if y_std < 1e-12:
            y_std = 1.0
        self.y_std = y_std
        y_centered=(y_train - mu_y) / y_std
        alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, y_centered))
        mu_norm = covXx @ alpha
        mu = mu_y + y_std * mu_norm
        return mu

    def distributions(self,data, X_new, sigma_kernel):
        y_train, x_train = data
        covXx = self.rbf_kernel(X_new, x_train, sigma=sigma_kernel, l=1)
        covXX = self.rbf_kernel(X_new, X_new, sigma=sigma_kernel, l=1)
        v = np.linalg.solve(self.L, covXx.T)
        var = covXX - v.T @ v
        sigma = np.sqrt(np.maximum(np.diag(var), 0))
        sigma = self.y_std * sigma
        return sigma

    def denormalize(self, x_norm):
        return x_norm * (self.X_max - self.X_min) + self.X_min

    def pairwise_distances(self,X):
        """
        X: np.array (N_points x D)
        Возвращает матрицу расстояний N x N
        """
        diff = X[:, None, :] - X[None, :, :]  # размерность N x N x D
        sqdist = np.sum(diff ** 2, axis=2)  # квадрат евклидова расстояния
        dist = np.sqrt(sqdist)
        return dist

    def plot_gp(self, X, visited_idx, y_train, param_names, mu, sigma):
        """
        X : np.array, нормализованная сетка (N_points x D)
        visited_idx : list of int, индексы реально вычисленных точек
        y_train : list or np.array, реальные значения функции для visited_idx
        param_names : list[str], имена параметров
        mu : np.array, GP предсказание для всей X
        sigma : np.array, GP uncertainty для всей X
        """
        D = X.shape[1]  # число параметров
        X_real = np.array([self.denormalize(X[i]) for i in visited_idx])
        Y_real = np.array(y_train)

        if D == 1:
            # --- 1D case ---
            plt.figure(figsize=(6, 4))
            plt.plot(self.denormalize(X).flatten(), mu, label="GP mean")
            plt.fill_between(self.denormalize(X).flatten(),
                             mu - sigma, mu + sigma,
                             alpha=0.2, label="GP ± sigma")
            plt.scatter(X_real.flatten(), Y_real, color="red", label="real points")
            plt.xlabel(param_names[0])
            plt.ylabel("Penalty")
            plt.legend()
            plt.show()

        elif D == 2:
            # --- 2D case ---
            # Создаём сетку по уникальным значениям параметров
            param1_vals = sorted(set([v[0] for v in self.denormalize(X)]))
            param2_vals = sorted(set([v[1] for v in self.denormalize(X)]))
            X1_grid, X2_grid = np.meshgrid(param1_vals, param2_vals)

            # Приводим GP предсказания к форме сетки
            mu_grid = mu.reshape(X1_grid.shape)
            sigma_grid = sigma.reshape(X1_grid.shape)

            # --- GP mean heatmap ---
            plt.figure(figsize=(6, 5))
            plt.contourf(X1_grid, X2_grid, mu_grid, levels=50, cmap='viridis')
            plt.colorbar(label='GP mean')
            plt.scatter(X_real[:, 0], X_real[:, 1], c=Y_real, edgecolors='red', label='real points')
            plt.xlabel(param_names[0])
            plt.ylabel(param_names[1])
            plt.legend()
            plt.show()

            # --- GP uncertainty heatmap ---
            plt.figure(figsize=(6, 5))
            plt.contourf(X1_grid, X2_grid, sigma_grid, levels=50, cmap='Reds')
            plt.colorbar(label='GP sigma')
            plt.scatter(X_real[:, 0], X_real[:, 1], c='black', edgecolors='white', label='real points')
            plt.xlabel(param_names[0])
            plt.ylabel(param_names[1])
            plt.legend()
            plt.show()
        else:
            print("Plotting for D > 2 is not supported. Consider slicing parameters or using projections.")

    def optimize(self, context, progress_queue):
        sim_result = SimulationResult()
        range_of_values = context.range_params.creating_a_range(None)
        param_names = list(range_of_values.keys())

        grid = list(product(*range_of_values.values()))
        X = np.array(grid, dtype=float)


        self.X_min = np.array([min(v) for v in range_of_values.values()])
        self.X_max = np.array([max(v) for v in range_of_values.values()])

        X = (X - self.X_min) / (self.X_max - self.X_min)
        D = self.pairwise_distances(X)
        sigma_kernel = np.median(D)
        visited_idx = []
        y_train = []

        # ---- first point ----
        first_idx = np.random.randint(len(X))
        visited_idx.append(first_idx)

        first_x = X[first_idx]
        first_x_real = self.denormalize(first_x)

        first_y = self.func(first_x_real, param_names, context, sim_result)

        y_train.append(first_y)

        for i in range(100):
            X_train = X[visited_idx]
            y_train_arr = np.array(y_train)

            data = [y_train_arr, X_train]

            mu = self.baesian(data, X, sigma_kernel)
            sigma = self.distributions(data, X, sigma_kernel)
            print("данные",mu, sigma)
            lcb = self.LCB(mu, sigma)

            # ---- choose only unexplored ----
            all_idx = np.arange(len(X))
            available = np.setdiff1d(all_idx, visited_idx)
            if len(available) == 0:
                print("All grid explored")
                break

            candidate_lcb = lcb[available]

            idx_local = np.argmin(candidate_lcb)
            idx_global = available[idx_local]

            next_x = X[idx_global]

            next_x_real = self.denormalize(next_x)
            next_y = self.func(next_x_real, param_names, context, sim_result)

            visited_idx.append(idx_global)
            y_train.append(next_y)

            print("next_x:", self.denormalize(X_train), self.denormalize(next_x), "y:", y_train)


            if len(y_train) > 1:
                self.delta = abs(
                    (y_train[-1] - y_train[-2]) /
                    max(abs(y_train[-2]), 1e-3)
                )

            # ---- finish ----
        best_idx = visited_idx[np.argmin(y_train)]
        print(best_idx)
        print(np.argmin(y_train))
        best_x_real = self.denormalize(X[best_idx])
        best_params = self.vector_to_params(best_x_real, param_names)
        print("лучшее",best_params, min(y_train))
        self.plot_gp(X, visited_idx, y_train, param_names, mu, sigma)
        if context.best_params is not None:
            context.runner.calculation(context.script_processor.build(best_params))
        else: print("the parameters are not optimized")
        progress_queue.put(("finished", None))

"""class Bayesian_optimization():
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

    def UCB(self,mean, sigma, *, b=2):

        return mean - b * sigma

    def plot(self,X, Y):
        plt.plot(X, Y)
        plt.show()

    def rbf_kernel(self,x_predict, x_init, sigma=1, l=0.3):  # матрица x1*x2

        x_predict = np.atleast_1d(x_predict)
        x_init = np.atleast_1d(x_init)
        value = sigma ** 2 * np.exp(-(x_predict[:, None] - x_init[None, :]) ** 2 / (2 * l ** 2))
        return value

    def inv(self, K, b, jitter=1e-8):
        K = K.copy()
        K += jitter * np.eye(len(K))
        return np.linalg.solve(K, b)

    def baesian(self, data, X_new):
        y_train, x_train = data
        covXx = self.rbf_kernel(X_new, x_train)
        covxx = self.rbf_kernel(x_train, x_train)
        noise = 1e-3
        covxx += np.eye(len(covxx)) * noise
        mu_y = np.mean(y_train)
        y_std=np.std(y_train)
        if y_std < 1e-12:
            y_std = 1.0
        self.y_std = y_std
        y_centered=(y_train - mu_y) / y_std
        #y_centered = y_train - np.full(len(y_train), float(mu_y))
        qw = self.inv(covxx, y_centered)
        mu_norm = covXx @ qw
        mu = mu_y + y_std * mu_norm
        return mu

    def distributions(self,data, X_new):
        y_train, x_train = data
        covXx = self.rbf_kernel(X_new, x_train)
        covxx = self.rbf_kernel(x_train, x_train)
        noise = 1e-3
        covxx += noise * np.eye(len(covxx))
        covxX = self.rbf_kernel(x_train, X_new)
        covXX = self.rbf_kernel(X_new, X_new)
        qw = self.inv(covxx, covxX)
        sigma = covXX - covXx @ qw
        sigma = np.sqrt(np.abs(np.diag(sigma)))
        sigma = self.y_std * sigma
        return sigma

    def optimize(self, context, progress_queue):
        sim_result = SimulationResult()
        range_of_values = context.range_params.creating_a_range(None)
        new_params = {}

        x = np.asarray(range_of_values.get("r"), dtype=float)
        data = [[], []]
        first_x = x[-10]
        first_y = self.func(first_x, context,sim_result)
        data[0].append(first_y)
        data[1].append(first_x)
        for i in range(20):
            temporary_data = []
            temporary_y = self.baesian(data, x)
            temporary_data.append(temporary_y)
            temporary_data.append(x)
            sigma = self.distributions(data, x)
            temporary_UCB = self.UCB(temporary_y, sigma)
            #plt.plot(data[1], data[0], label="real")
            plt.plot(x, temporary_data[0])
            plt.plot(x, sigma, label="sigma")
            plt.ylim(-2, 8)
            visited = np.array(data[1])
            mask = ~np.isin(x, visited)
            next_x = x[mask][np.argmin(temporary_UCB[mask])]
            next_y = self.func(next_x,  context,sim_result)
            print("x:", data[1], "y:", data[0])
            print("x:", next_x, "y:", next_y)
            data[0].append(next_y)
            data[1].append(next_x)
            plt.legend()
            plt.show()
            self.delta = abs((data[0][-1] - data[0][-2]) / max(data[0][-2], 1e-3))
        last_y = self.func(data[1][-1], context, sim_result)
        if context.best_params is not None:
            context.runner.calculation(context.script_processor.build(context.best_params))
        else: print("the parameters are not optimized")
        progress_queue.put(("finished", None))"""




