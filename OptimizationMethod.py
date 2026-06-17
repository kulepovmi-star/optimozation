from abc import ABC, abstractmethod
import random
from re import search
import math
import matplotlib.pyplot as plt
from itertools import product
import numpy as np
from collections import defaultdict
from typing import Dict

import torch
from torch import double, tensor, cat, norm
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize
from botorch.acquisition import UpperConfidenceBound
from gpytorch.kernels import RBFKernel, ScaleKernel
from botorch.models.transforms.outcome import Standardize
from simulation_result import SimulationResult
from gpytorch.constraints import GreaterThan
from gpytorch.likelihoods import GaussianLikelihood
from botorch.acquisition import ExpectedImprovement


class OptimizationMethod(ABC):
    def __init__(self, iterations):
        self.iterations = iterations


    @abstractmethod
    def optimize(self, context, progress_queue):
        """основная задача методов оптимизации заключается в минимизации целевого параметра. В данный момент между искомыми параметрами не выбирается лучший, для каждой итерации,
        а шаг за шагом применяется каждый параметр к расчетной модели, отвечающий цели. """
        pass

    def calculation(self, sim_result, context, params):
        print("устойчивость",context.constraints.get("buckling"))
        print(params)
        context.runner.calculation_static(context.script_processor.build({**params}))
        sim_result.save_data_static(base_dir=context.base_dir)
        if context.constraints.get("buckling"):
            context.runner.calculation_buckling(context.script_processor.build({**params}))
            sim_result.save_data_buckling(base_dir=context.base_dir)
        penalty=context.objective.evaluate(sim_result, context, {**params})
        print("penalty", penalty)
        return float(penalty)


class Step_by_step_change(OptimizationMethod):
    def __init__(self, iterations=0,checkbox=False):
        super().__init__(iterations)
        self.params={}
        self.checkbox=checkbox


    def substitution(self, sim_result, context, range_of_values, progress_queue,*, name):
        iteration=0
        keys=range_of_values.keys()
        values=range_of_values.values()
        print(self.checkbox)
        if self.checkbox:
            iterable=list(product(*values))
            print(iterable)
        else:
            iterable = list(zip(*values))

        iterations=len(iterable)
        for combo in iterable:
            iteration+=1
            params = dict(zip(keys, combo))
            self.calculation(sim_result, context, params)
            progress = int(iteration / iterations * 100)
            progress_queue.put(("progress", progress))

    def optimize(self, context, progress_queue):
        range_of_values = context.range_params.creating_a_range(None)
        print(range_of_values)
        sim_result = SimulationResult()

        name_params = iter(range_of_values.keys())

        self.substitution(sim_result, context, range_of_values, progress_queue, name=name_params)
        if context.best_params is not None:
            context.runner.calculation_static(context.script_processor.build(context.best_params))
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
                penalty = self.calculation(sim_result, context, new_params)
                print("параметры", context.best_params)
                print(iteration, "итерации")

                iteration += 1
                progress = int((iteration + 1) / self.iterations * 100)
                progress_queue.put(("progress", progress))
            if phase < phases - 1 and context.best_params is not None:
                range_of_values = context.range_params.creating_a_range(context.best_params)

        if context.best_params is not None:
            context.runner.calculation_static(context.script_processor.build(context.best_params))
        else: print("the parameters are not optimized")
        progress_queue.put(("finished", None))

class GradientDescent(OptimizationMethod):

    epsilon = 1e-6

    def __init__(self, iterations, *,steps= 0.01, learning_rate=0.04, b1=0.9, b2=0.99):
        super().__init__(iterations)
        self.step_size = steps
        self.lr = learning_rate
        self.b1=b1
        self.b2=b2
        print(steps, learning_rate, b1)

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
            penalty_base = self.calculation(sim_result, context, new_params)

            gradients = {}

            # 2 вычисляем все градиенты

            for key, value in new_params.items():
                step = value * dict_step[key]

                max_value = value + step

                if max_value > max(range_of_values[key]):
                    max_value = max(range_of_values[key])
                params_plus = {**new_params, **{key: max_value}}
                penalty_plus = self.calculation(sim_result, context, params_plus)
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
            context.runner.calculation_static(context.script_processor.build(context.best_params))
        else: print("the parameters are not optimized")
        progress_queue.put(("finished", None))


class Bayesian_optimization(OptimizationMethod):

    def __init__(self, iterations, selection, b):
        super().__init__(iterations)
        self.iterations = iterations
        self.delta=float("inf")
        self.L=None
        self.selection=selection
        self.b=b
        self.sigma=1
        self.l=1
        self.best_params=None

    def random_search(self, number_of_points, range_of_values, context, sim_result, param_names):
        X_train, Y_train = [], []
        keys = range_of_values.keys()
        for _ in range(number_of_points**len(keys)):
            new_params = {
                k: random.choice(v)
                for k, v in range_of_values.items()
            }
            X_train.append(list(new_params.values()))
            params = self.vector_to_params(X_train[-1], param_names)
            penalty = self.calculation( sim_result, context, params)
            Y_train.append(
                penalty
            )

        Y_train = np.array(Y_train, float)
        print(np.array(X_train, float), np.array(Y_train, float))
        return np.array(X_train, float), np.array(Y_train, float)




    def vector_to_params(self, x_vec, param_names):
        return dict(zip(param_names, x_vec))


    def plot_gp(self, model, bounds, train_x_real, train_y, param_names):
        print(bounds)
        """
        X : np.array, нормализованная сетка (N_points x D)
        visited_idx : list of int, индексы реально вычисленных точек
        y_train : list or np.array, реальные значения функции для visited_idx
        param_names : list[str], имена параметров
        mu : np.array, GP предсказание для всей X
        sigma : np.array, GP uncertainty для всей X
        """
        D = len(bounds[0])

        if D == 1:
            X_real = np.linspace(
                bounds[0][0].item(),
                bounds[1][0].item(),
                1000
            ).reshape(-1, 1)

            X_tensor_real = tensor(
                X_real,
                dtype=double
            )
            X_tensor = normalize(
                X_tensor_real,
                bounds
            )
            visited_x = train_x_real.detach().numpy()
            y_train_np = train_y.detach().numpy().flatten()
            posterior = model.posterior(X_tensor)
            mu = posterior.mean.detach().numpy().flatten()

            sigma = (
                posterior.variance
                .sqrt()
                .detach()
                .numpy()
                .flatten()
            )
            # --- 1D case ---
            plt.figure(figsize=(6, 4))
            plt.plot(X_real.flatten(), mu, label="GP mean")
            plt.fill_between(X_real.flatten(),
                             mu - sigma, mu + sigma,
                             alpha=0.2, label="GP ± sigma")
            plt.scatter(visited_x, y_train_np, color="red", label="real points")
            plt.xlabel(param_names[0])
            plt.ylabel(f"f({param_names[0]})")
            plt.legend()
            plt.show()
        #
        elif D == 2:
            x = np.linspace(
                bounds[0][0].item(),
                bounds[1][0].item(),
                100
            )

            y = np.linspace(
                bounds[0][1].item(),
                bounds[1][1].item(),
                100
            )

            X_grid, Y_grid = np.meshgrid(x, y)

            X = np.column_stack([
                X_grid.ravel(),
                Y_grid.ravel()
            ])

            X_tensor_real = tensor(
                X,
                dtype=double
            )

            X_tensor = normalize(
                X_tensor_real,
                bounds
            )

            posterior = model.posterior(X_tensor)

            mu = (
                posterior.mean
                .detach()
                .numpy()
                .reshape(X_grid.shape)
            )

            sigma = (
                posterior.variance
                .sqrt()
                .detach()
                .numpy()
                .reshape(X_grid.shape)
            )

            visited_x = train_x_real.detach().numpy()

            y_train_np = (
                train_y
                .detach()
                .numpy()
                .flatten()
            )


            # --- GP mean heatmap ---
            plt.figure(figsize=(6, 5))
            plt.contourf(x, y, mu, levels=150, cmap='viridis')
            plt.colorbar(label='GP mean')
            plt.scatter(visited_x[:, 0], visited_x[:, 1], c=y_train_np, edgecolors='red', label='real points')
            plt.xlabel(param_names[0])
            plt.ylabel(param_names[1])
            plt.legend()
            plt.show()

            # --- GP uncertainty heatmap ---
            plt.figure(figsize=(6, 5))
            plt.contourf(x, y, sigma, levels=150, cmap='viridis')
            plt.colorbar(label='GP sigma')
            plt.scatter(visited_x[:, 0], visited_x[:, 1], c='black', edgecolors='white', label='real points')
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
        train_x_np, train_y_np = self.random_search(self.selection, range_of_values, context, sim_result, param_names)

        low_bound=[]
        high_bound = []
        for k, v in range_of_values.items():
            low_bound.append(min(v))
            high_bound.append(max(v))

        bounds = tensor(
            [
                low_bound,  # нижняя граница
                high_bound  # верхняя граница
            ],
            dtype=double
        )

        train_x_real  = tensor(
            train_x_np,
            dtype=double
        )

        train_y = tensor(
            train_y_np,
            dtype=double
        ).unsqueeze(-1)
        likelihood = GaussianLikelihood(
            noise_constraint=GreaterThan(1e-3)
        )
        for i in range(self.iterations):
            train_x = normalize(
                train_x_real,
                bounds)
            model = SingleTaskGP(train_x, train_y,
                                likelihood=likelihood,
                                 covar_module=ScaleKernel(
                                     RBFKernel()),
                                 outcome_transform=Standardize(m=1)
                                 )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            # acq = UpperConfidenceBound(
            #     model=model,
            #     beta=self.b,
            #     maximize=False
            # )
            best_f = train_y.min()

            acq = ExpectedImprovement(
                model=model,
                best_f=best_f,
                maximize=False
            )
            normalized_bounds = tensor(
                [
                    [0.0]*len(range_of_values.keys()),
                    [1.0]*len(range_of_values.keys())
                ],
                dtype=double
            )

            candidate_normalized, acq_value = optimize_acqf(
                acq_function=acq,

                bounds=normalized_bounds,

                q=1,

                num_restarts=10,

                raw_samples=100
            )

            candidate_real = unnormalize(
                candidate_normalized,
                bounds
            )
            dists = norm(train_x_real - candidate_real, dim=1)

            if torch.min(dists) < 1e-3:
                print("Point too close, skipping")
                for _ in range(10):

                    dists = torch.norm(
                        train_x_real - candidate_real,
                        dim=1
                    )

                    if torch.min(dists) > 1e-3:
                        break

                    candidate_normalized = torch.rand_like(
                        candidate_normalized
                    )

                    candidate_real = unnormalize(
                        candidate_normalized,
                        bounds
                    )

            params = self.vector_to_params(candidate_real.detach().numpy().flatten(), param_names)
            new_y = self.calculation(sim_result, context, params )
            train_x_real = cat([
                train_x_real,
                candidate_real
            ], dim=0)
            new_y_tensor = tensor(
                [[new_y]],
                dtype=double
            )

            train_y = cat([
                train_y,
                new_y_tensor
            ], dim=0)
            progress = int((i + 1) / self.iterations * 100)
            progress_queue.put(("progress", progress))



        if context.best_params is not None:
            print(context.best_params)
            context.runner.calculation_static(context.script_processor.build(context.best_params))
        else: print("the parameters are not optimized")
        progress_queue.put(("finished", None))
        self.plot_gp(model,
                     bounds, train_x_real, train_y, param_names
                     )








