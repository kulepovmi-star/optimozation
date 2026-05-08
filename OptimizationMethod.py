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

class Step_by_step_change(OptimizationMethod):
    def __init__(self,  steps, iterations=0,):
        super().__init__(iterations)
        self.params={}
        self.steps=steps


    def calculation(self, sim_result, context, params):
        print(params)
        # context.runner.calculation(context.script_processor.build({**params}))
        # sim_result.save_data(base_dir=context.base_dir)
        # context.objective.evaluate(sim_result, context, {**params})

    def substitution(self, sim_result, context, range_of_values, *, name):
        keys=range_of_values.keys()
        values=range_of_values.values()

        for combo in product(*values):
            params=dict(zip(keys, combo))
            self.calculation(sim_result, context, params)


    def optimize(self, context, progress_queue):
        sim_result = SimulationResult()
        range_of_values = context.range_params.range_by_step(self.steps)
        name_params = iter(range_of_values.keys())

        self.substitution(sim_result, context, range_of_values, name=name_params)
        if context.best_params is not None:
            context.runner.calculation(context.script_processor.build(context.best_params))
        else: print("the parameters are not optimized")
        progress_queue.put(("finished", None))




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

    def __init__(self, iterations, *,steps= 0.01, l_r=0.04, b1=0.9, b2=0.99):
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
            new_params[key] = random.choice(value)

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

            penalty_base = np.log(context.objective.evaluate(
                sim_result, context, new_params
            )+1)

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

                penalty_plus = np.log(context.objective.evaluate(
                    sim_result, context, params_plus
                )+1)

                gradient = (penalty_plus - penalty_base) / step
                gradient=max(-5, min(5, gradient))
                gradients[key] = gradient

                gradient_dict[key].append(gradient)



            # 3 обновляем параметры

            temporal_params = {}

            for key, value in new_params.items():

                gradient = gradients[key]

                if len(gradient_dict[key])>1 and np.sign(gradient_dict[key][-1]) != np.sign(gradient_dict[key][-2]):
                    dict_lr[key] = dict_lr[key] / 1

                v[key] = self.b1 * v[key] + (1 - self.b1) * gradient
                G[key] = self.b2 * G[key] + (1 - self.b2) * gradient ** 2
                print(
                    key,
                    "grad:", gradient_dict[key][-1],
                    "v[key]", v[key],
                    "G[key]", G[key],
                )

                v_correction=v[key]/(1-self.b1**iteration)
                G_correction=G[key]/(1-self.b2**iteration)

                value_norm = (
                                     value - min(range_of_values[key])
                             ) / (
                                     max(range_of_values[key]) - min(range_of_values[key])
                             )

                new_value_norm = value_norm - dict_lr[key] * v_correction / (G_correction + self.epsilon) ** 0.5

                new_value = new_value_norm * (
                        max(range_of_values[key]) - min(range_of_values[key])
                ) + min(range_of_values[key])

                new_value = max(
                    value - 0.2 * value,
                    min(value + 0.2 * value, new_value)
                )
                params.append(new_value)
                #print("величина", params)
                if new_value<min(range_of_values[key]):
                    new_value=min(range_of_values[key])
                if new_value>max(range_of_values[key]):
                    new_value=max(range_of_values[key])
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
    def __init__(self, iterations):
        self.iterations = iterations
        self.delta=float("inf")
        self.L=None
        self.sigma=1
        self.l=1
    def random_search(self, number_of_points, range_of_values, context, sim_result):
        X_train, Y_train=[],[]
        for _ in range(number_of_points):
            new_params = {
                k: random.choice(v)
                for k, v in range_of_values.items()
            }
            X_train.append(list(new_params.values()))
            context.runner.calculation(context.script_processor.build({**new_params}))
            sim_result.save_data(base_dir=context.base_dir)

            Y_train.append(
                np.log(context.objective.evaluate(
                    sim_result, context, new_params
                )+1)
            )
        Y_train = np.array(Y_train, float)
        Y_train = np.clip(Y_train, 0, 20)
        return np.array(X_train, float), np.array(Y_train, float)

    def leave_one_out(self, X_train, Y_train):
        std = np.std(X_train, axis=0)
        std[std < 1e-12] = 1.0
        Xn = (X_train - np.mean(X_train, axis=0)) / std
        Yn = (Y_train - np.mean(Y_train)) / np.std(Y_train)
        preds=[]
        N = len(Xn)
        for i in range(N):
            mask = np.ones(N, dtype=bool)
            mask[i] = False

            X_sub = Xn[mask]
            Y_sub = Yn[mask]

            data = [Y_sub, X_sub]

            mu = self.baesian(data, Xn[i])

            preds.append(mu[0])

        return self.MSE(Yn, np.array(preds))

    def MSE(self, y_true, y_pred):
        y_true = np.ravel(y_true)
        y_pred = np.ravel(y_pred)
        return np.mean((y_true-y_pred)**2)

    def log_marginal_likelihood(self, X, y):
        K = self.rbf_kernel(X, X) + 1e-6 * np.eye(len(X))
        L = np.linalg.cholesky(K)

        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))

        term1 = -0.5 * y.T @ alpha
        term2 = -np.sum(np.log(np.diag(L)))
        term3 = -0.5 * len(X) * np.log(2 * np.pi)

        return (term1 + term2 + term3).item()
    def hyperparams_random_search(self, number_of_points, range_of_values, context, sim_result, sample, sigma_params, l_params):
        error=[]
        params=[]
        print("диапазон",range_of_values)
        sigma_grid = np.linspace(sigma_params[0], sigma_params[1], 100)
        l_grid = np.linspace(l_params[0], l_params[1], 100)
        X_train, Y_train = self.random_search(number_of_points, range_of_values, context, sim_result)
        for sigma in sigma_grid:
            for l in l_grid:
                self.sigma = sigma
                self.l = l
                print( " sigma",self.sigma, " l",self.l)
                params.append((self.sigma, self.l))
                error.append(self.log_marginal_likelihood(X_train, Y_train))
        best = np.argmax(error)
        self.sigma, self.l = params[best]
        print("лучшее", params[best])
        return X_train, Y_train

    def vector_to_params(self, x_vec, param_names):
        return dict(zip(param_names, x_vec))

    def func(self, x_vec, param_names, context, sim_result):
        params = self.vector_to_params(x_vec, param_names)

        context.runner.calculation(
            context.script_processor.build(params)
        )

        sim_result.save_data(base_dir=context.base_dir)

        penalty = context.objective.evaluate(
            sim_result, context, params
        )
        print("penalty:", penalty)
        penalty = np.log(penalty + 1)
        return penalty


    def LCB(self,mean, sigma, *, b=0.5):
        return mean - b * sigma


    def rbf_kernel(self,x_predict, x_init):  # матрица x1*x2
        """если точки кучкуются слишком рано → l слишком большой
        если прыгает хаотично → l слишком маленький
        если игнорирует хорошие точки → σ слишком маленький"""
        X1 = np.atleast_2d(x_predict)
        X2 = np.atleast_2d(x_init)

        diff = X1[:, None, :] - X2[None, :, :]
        sqdist = np.sum(diff ** 2, axis=2)

        return self.sigma ** 2 * np.exp(-sqdist / (2 * self.l ** 2))


    def baesian(self, data, X_new):
        y_train, x_train = data
        covXx = self.rbf_kernel(X_new, x_train)
        covxx = self.rbf_kernel(x_train, x_train)
        noise = 1e-6
        K = covxx + noise * np.eye(len(covxx))
        self.L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(self.L.T,
                                np.linalg.solve(self.L, y_train))

        mu = covXx @ alpha

        return mu  # уже нормализованный!

    def distributions(self,data, X_new):
        y_train, x_train = data
        covXx = self.rbf_kernel(X_new, x_train)
        covXX = self.rbf_kernel(X_new, X_new)
        v = np.linalg.solve(self.L, covXx.T)
        var = covXX - v.T @ v
        sigma = np.sqrt(np.maximum(np.diag(var), 0))

        return sigma  # тоже в нормализованном масштабе

    def denormalize(self, x_norm):
        return x_norm * self.X_std + self.mean_X

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

    def normalization(self, X, fit=False):

        if fit:
            self.mean_X = np.mean(X, axis=0)
            self.X_std = np.std(X, axis=0)
            self.X_std[self.X_std < 1e-12] = 1.0

        return (X - self.mean_X) / self.X_std

    def optimize(self, context, progress_queue):

        sim_result = SimulationResult()
        range_of_values = context.range_params.creating_a_range(None)
        param_names = list(range_of_values.keys())
        grid = list(product(*range_of_values.values()))
        X = np.array(grid, dtype=float)
        X = self.normalization(X, fit=True)
        X_train, Y_train = self.hyperparams_random_search(15, range_of_values, context, sim_result, 100, [3, 15],
                                                          [0.3, 5])
        X_train_rs = self.normalization(X_train, fit=False)
        Y_train_rs = list(Y_train)

        visited_idx = []
        y_train = []
        print("начало алгоритма")
        for x_rs, y_rs in zip(X_train_rs, Y_train_rs):

            # ищем ближайшую точку сетки
            idx = np.argmin(np.linalg.norm(X - x_rs, axis=1))

            if idx not in visited_idx:
                visited_idx.append(idx)
                y_train.append(y_rs)

        # ---- first point ----
        first_idx = np.random.randint(len(X))
        visited_idx.append(first_idx)

        first_x = X[first_idx]
        first_x_real = self.denormalize(first_x)

        first_y = self.func(first_x_real, param_names, context, sim_result)

        y_train.append(first_y)
        for iteration in range(self.iterations):
            X_train = X[visited_idx]
            y_train_arr = np.array(y_train)

            y_mean = np.mean(y_train_arr)
            y_std = np.std(y_train_arr)
            if y_std < 1e-12:
                y_std = 1.0
            y_train_norm = (y_train_arr - y_mean) / y_std

            data = [y_train_norm, X_train]
            mu = self.baesian(data, X)
            sigma = self.distributions(data, X)
            mu = y_mean + y_std * mu
            sigma = y_std * sigma

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
            progress = int((iteration + 1) / self.iterations * 100)
            progress_queue.put(("progress", progress))
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








