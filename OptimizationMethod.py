from abc import ABC, abstractmethod
import random
from re import search
import math


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
                print(trial, "итерации")

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
            progress = int((iteration + 1) / self.iterations * 100)
            progress_queue.put(("progress", progress))

        if context.best_params is not None:
            context.runner.calculation(context.script_processor.build(context.best_params))
        else: print("the parameters are not optimized")
        progress_queue.put(("finished", None))
    """def optimize(self, context):
        params=[]
        penalty_dict = {}
        gradient_dict = {}
        dict_grad={}
        dict_step={}
        dict_lr={}

        stop_flag_dict={}
        sim_result = SimulationResult()
        # случайный выбор исходных параметров
        new_params = {}
        print("step",self.steps)
        range_of_values = context.range_params.creating_a_range(None)
        v = dict()
        G = dict()
        for key, value in range_of_values.items():
            mean_value = np.mean(value)
            new_params[key] = mean_value
            stop_flag_dict[key] = False
            dict_grad[key]=[]
            dict_step[key]=self.steps.get(key, self.step_size)
            dict_lr[key] = self.l_r.get(key, self.lr)
            v[key]=0
            G[key]=0
            penalty_dict[key] = []
            gradient_dict[key]=[]
        context.runner.calculation(context.script_processor.build(new_params))
        sim_result.save_data(base_dir=context.base_dir)
        for key in range_of_values:
            print(key)
            print(type(penalty_dict[key]))
            penalty_dict[key].append(context.objective.evaluate(sim_result, context, new_params))

        # условие остановки итерации 1) неправильная логика при смене знака, так и макс, градиент равен 0 или около 0


        iteration=0
        while list(stop_flag_dict.values()).count(True) != len(stop_flag_dict.values()) or iteration < self.iterations:
            temporal_params={}
            for key, value in new_params.items():
                iteration += 1
                print("итерации ", iteration)
                if stop_flag_dict[key] == False:

                    # если во время итерации мы вышли за диапазон то возвращаем исходные значения
                    if min(range_of_values[key]) > value:
                        value = min(range_of_values[key])
                        dict_lr[key] = self.l_r.get(key, self.lr)
                        dict_step[key] = self.steps.get(key, self.step_size)

                    max_value = value + value * dict_step[key]

                    if max(range_of_values[key]) < max_value:
                        max_value = max(range_of_values[key])

                    print("параметры величин", key, value, "+", max_value,)
                    value_norm = (value - min(range_of_values[key])) / (
                                max(range_of_values[key]) - min(range_of_values[key]))
                    context.runner.calculation(context.script_processor.build({**new_params, **{key: max_value}}))
                    sim_result.save_data(base_dir=context.base_dir)
                    result_plus = context.objective.evaluate(sim_result, context, {**new_params, **{key: max_value}})

                    gradient = (result_plus - penalty_dict[key][-1]) / (dict_step[key])
                    gradient_dict[key].append(gradient)
                    penalty_dict[key].append(result_plus)

                    # при развороте меняем шаг
                    if len(gradient_dict[key])>1 and numpy.sign(gradient_dict[key][-1]) != numpy.sign(gradient_dict[key][-2]):
                        print("меняем шаг")
                        dict_lr[key] = dict_lr[key] / 1.1
                    # ограничение на градиент
                    # gradient = max(-1.5, min(1.5, gradient))

                    print("параметры", key, "+:", max_value, result_plus, penalty_dict[key][-2], "lr:",
                          dict_lr[key], "grad:", gradient)

                    if abs(gradient) < 0.04:  # или какому-нибудь небольшому значению
                        stop_flag_dict[key] = True

                    v[key] = self.b1 * v[key] + (1 - self.b1) * gradient
                    G[key] = self.b2 * G[key] + (1 - self.b2) * gradient ** 2
                    new_value_norm = value_norm - dict_lr[key] * v[key] / (G[key] + self.epsilon) ** 0.5
                    print("изменение", dict_lr[key] * v[key] / (G[key] + self.epsilon))

                    dict_grad[key].append(gradient)
                    temporal_params[key] = max(value - 0.2 * value, min(value + 0.2 * value, new_value_norm * (
                                max(range_of_values[key]) - min(range_of_values[key])) + min(range_of_values[key])))
                    params.append(float(temporal_params[key]))
                    print("Результат", temporal_params[key], params)

                else:
                    if context.best_params is not None:
                        context.runner.calculation(context.script_processor.build(context.best_params))
                    else:
                        print("the parameters are not optimized")
                    return None
            new_params={**new_params, **temporal_params}
        if context.best_params is not None:
            context.runner.calculation(context.script_processor.build(context.best_params))
            return None
        else:
            print("the parameters are not optimized")
            return None
        while list(stop_flag_dict.values()).count(True)!=len(stop_flag_dict.values()) or iteration<self.iterations:
            for key, value in new_params.items():
                iteration+=1
                print("итерации ", iteration)
                if stop_flag_dict[key] == False :

                    # если во время итерации мы вышли за диапазон то возвращаем исходные значения
                    if min(range_of_values[key]) > value:
                        value = min(range_of_values[key])
                        dict_lr[key] = self.l_r.get(key, self.lr)
                        dict_step[key] = self.steps.get(key, self.step_size)

                    max_value = value + value*dict_step[key]


                    if max(range_of_values[key])<max_value:
                        max_value=max(range_of_values[key])

                    min_value = value - value * dict_step[key]
                    if min(range_of_values[key]) > min_value:
                        min_value = min(range_of_values[key])

                    print("параметры величин", key, value, "+", max_value, "-",min_value,)
                    value_norm=(value-min(range_of_values[key]))/(max(range_of_values[key])-min(range_of_values[key]))
                    context.runner.calculation(context.script_processor.build({**new_params, **{key:max_value}}))
                    sim_result.save_data(base_dir=context.base_dir)
                    result_plus = context.objective.evaluate(sim_result, context, {**new_params, **{key:max_value}})

                    context.runner.calculation(context.script_processor.build({**new_params, **{key:min_value}}))
                    sim_result.save_data(base_dir=context.base_dir)
                    result_minus = context.objective.evaluate(sim_result, context,  {**new_params, **{key:min_value}})



                    gradient=(result_plus - result_minus)/(2*dict_step[key])
                    penalty_list.append(gradient)

                    # при развороте меняем шаг
                    if len(penalty_list)>1 and numpy.sign(penalty_list[-1])!=numpy.sign(penalty_list[-2]):
                        print("меняем шаг")
                        dict_lr[key]=dict_lr[key]/2
                    # ограничение на градиент
                    #gradient = max(-1.5, min(1.5, gradient))

                    print("параметры", key, "+:", max_value, "-:",min_value, result_plus, result_minus, "delta:", dict_step[key], "grad:", gradient)

                    if abs(result_plus - result_minus) < 0.004: # или какому-нибудь небольшому значению
                        stop_flag_dict[key] = True

                    v[key]=self.b1*v[key]+(1-self.b1)*gradient
                    G[key]=self.b2*G[key]+(1-self.b2)*gradient**2
                    new_value_norm=value_norm - dict_lr[key] * v[key]/(G[key]+self.epsilon)**0.5
                    print("изменение", dict_lr[key]*v[key]/(G[key]+self.epsilon))

                    dict_grad[key].append(gradient)
                    new_params[key] = max(value-0.2*value, min(value+0.2*value, new_value_norm*(max(range_of_values[key])-min(range_of_values[key]))+min(range_of_values[key])))
                    print("Результат", new_params[key], )

                else:
                    if context.best_params is not None:
                        context.runner.calculation(context.script_processor.build(context.best_params))
                    else:
                        print("the parameters are not optimized")
                    return None


        if context.best_params is not None:
            context.runner.calculation(context.script_processor.build(context.best_params))
            return None
        else:
            print("the parameters are not optimized")
            return None"""




