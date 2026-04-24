from abc import ABC, abstractmethod
import numpy as np
import math

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from context import OptimizationContext
    from simulation_result import SimulationResult


class OptimizationFunction(ABC):
    best_value = float("inf")
    norm_mass = 0
    norm_stress = 0
    norm_strain = 0
    def __init__(self, k=100):
        self.k=k
    #меньшее значение - лучше
    @abstractmethod
    def evaluate(self, simulation_result: "SimulationResult", context:"OptimizationContext", best_params):
        """целевая функция возвращает параметр если он попал в целевое условие, иначе None"""
        pass


class Mass(OptimizationFunction):
    mass = []

    # для градиентного спуска нам необходимо работать только с penalty, поскольку на каждой итерации мы стремимся его уменьшить,
    # для работы с методом лучшей пробы нам необходимо записывать параметры проходящие через установленные ограничения как и в градиенте, но записывать только если значение массы, то есть penalty является наименьшим
    def evaluate(self, simulation_result: "SimulationResult", context:"OptimizationContext", best_params, ):
        max_stress_component = max(stress[6] for stress in simulation_result.stress_list)
        max_strain_component = max(max(strain) for strain in simulation_result.strain_list)
        delta_stress= max_stress_component / context.constraints.get("Stress", float("inf"))
        delta_disp = max_strain_component / context.constraints.get("Displacement", float("inf"))

        # приводим массу к порядку 1
        if not self.norm_mass:
            self.norm_mass=simulation_result.mass

        print("данные", "mass:",simulation_result.mass, "disp:",delta_disp, "stress",delta_stress)
        mass_ratio = (simulation_result.mass / self.norm_mass)**2
        self.mass.append(mass_ratio)
        print("mass_ratio", self.mass)

        def violation(r):
            return max(0.0, r - 1.0)

        constraint_penalty = (
                violation(delta_stress) ** 2 +
                violation(delta_disp) ** 2
        )

        penalty = mass_ratio * (1 + self.k * constraint_penalty)

        # сохраняем только допустимые решения
        if constraint_penalty == 0 and self.best_value > simulation_result.mass:
            print("записали")
            self.best_value = simulation_result.mass
            context.best_params = best_params

        return penalty



class Stress(OptimizationFunction):
    stress=[]
    def evaluate(self, simulation_result: "SimulationResult", context:"OptimizationContext", best_params, k=100):

        max_stress_component = max(stress[6] for stress in simulation_result.stress_list)
        max_strain_component = max(max(strain) for strain in simulation_result.strain_list)
        delta_mass=simulation_result.mass/context.constraints.get("Mass", float("inf"))
        delta_disp=max_strain_component/ context.constraints.get("Displacement", float("inf"))

        # приводим напряжения к порядку 1
        if not self.norm_stress:
            self.norm_stress = max_stress_component

        print("данные", "stress", max_stress_component, "mass:", delta_mass, "disp:", delta_disp, )
        stress_ratio = (max_stress_component / self.norm_stress) ** 2
        self.stress.append(stress_ratio)
        print("stress_ratio", self.stress)

        def violation(r):
            return max(0.0, r - 1.0)

        constraint_penalty = (
                violation(delta_mass) ** 2 +
                violation(delta_disp) ** 2
        )

        penalty = stress_ratio * (1 + k * constraint_penalty)

        # сохраняем только допустимые решения
        if constraint_penalty == 0 and self.best_value > max_stress_component:
            print("записали")
            self.best_value = max_stress_component
            context.best_params = best_params

        return penalty

class Strain(OptimizationFunction):
    strain=[]
    def evaluate(self, simulation_result: "SimulationResult", context:"OptimizationContext", best_params, k=100):
        max_stress_component = max(stress[6] for stress in simulation_result.stress_list)
        max_strain_component = max(max(strain) for strain in simulation_result.strain_list)
        delta_stress = max_stress_component / context.constraints.get("Stress", float("inf"))
        delta_mass=simulation_result.mass/context.constraints.get("Mass", float("inf"))

        # приводим напряжения к порядку 1
        if not self.norm_strain:
            self.norm_strain = max_strain_component

        print("данные","disp:", max_strain_component, "stress", delta_stress, "mass:", delta_mass,  )
        strain_ratio = (max_stress_component / self.norm_stress) ** 2
        self.strain.append(strain_ratio)
        print("strain_ratio", self.strain)

        def violation(r):
            return max(0.0, r - 1.0)

        constraint_penalty = (
                violation(delta_mass) ** 2 +
                violation(delta_stress) ** 2
        )

        penalty = strain_ratio * (1 + k * constraint_penalty)

        # сохраняем только допустимые решения
        if constraint_penalty == 0 and self.best_value > max_strain_component:
            print("записали")
            self.best_value = max_strain_component
            context.best_params = best_params

        return penalty