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
    #меньшее значение - лучше
    @abstractmethod
    def evaluate(self, simulation_result: "SimulationResult", context:"OptimizationContext", best_params):
        """целевая функция возвращает параметр если он попал в целевое условие, иначе None"""
        pass


class Mass(OptimizationFunction):
    # для градиентного спуска нам необходимо работать только с penalty, поскольку на каждой итерации мы стремимся его уменьшить,
    # для работы с методом лучшей пробы нам необходимо записывать параметры проходящие через установленные ограничения как и в градиенте, но записывать только если значение массы, то есть penalty является наименьшим
    def evaluate(self, simulation_result: "SimulationResult", context:"OptimizationContext", best_params, k=30):

        max_stress_component = max(stress[6] for stress in simulation_result.stress_list)
        max_strain_component = max(max(strain) for strain in simulation_result.strain_list)
        delta_stress= max_stress_component / context.constraints.get("Stress", float("inf"))
        delta_disp = max_strain_component / context.constraints.get("Displacement", float("inf"))

        # приводим массу к порядку 1
        if not self.norm_mass:
            self.norm_mass=simulation_result.mass

        print("данные", "mass:",simulation_result.mass, "disp:",delta_disp, "stress",delta_stress)
        penalty = simulation_result.mass/self.norm_mass
        print("penalty:", penalty)
        if 1 < delta_stress < float("inf"):
            penalty += k*(delta_stress-1)**2
            print("penalty:", penalty)

        if 1 < delta_disp < float("inf"):
            penalty += k*(delta_disp-1)**2

        if penalty == simulation_result.mass/self.norm_mass and self.best_value > simulation_result.mass:
            self.best_value = simulation_result.mass
            print("записали")
            print(best_params)
            context.best_params = best_params

        return penalty



class Stress(OptimizationFunction):
    def evaluate(self, simulation_result: "SimulationResult", context:"OptimizationContext", best_params):

        max_stress_component = max(stress[6] for stress in simulation_result.stress_list)
        max_strain_component = max(max(strain) for strain in simulation_result.strain_list)
        delta_mass=simulation_result.mass/context.constraints.get("Mass", float("inf"))
        delta_disp=max_strain_component/ context.constraints.get("Displacement", float("inf"))


        penalty = max_stress_component
        if delta_mass>1:
            penalty*=delta_mass

        if delta_disp>1:
            penalty*=delta_disp

        if penalty==max_stress_component and self.best_value > max_stress_component:
            self.best_value = max_stress_component
            context.best_params = best_params

        return penalty


class Strain(OptimizationFunction):
    def evaluate(self, simulation_result: "SimulationResult", context:"OptimizationContext", best_params):
        max_stress_component = max(stress[6] for stress in simulation_result.stress_list)
        max_strain_component = max(max(strain) for strain in simulation_result.strain_list)
        delta_stress = max_stress_component / context.constraints.get("Stress", float("inf"))
        delta_mass=simulation_result.mass/context.constraints.get("Mass", float("inf"))


        penalty = max_strain_component
        if delta_stress > 1:
            penalty *= delta_stress

        if delta_mass > 1:
            penalty *= delta_mass

        if penalty == max_strain_component and self.best_value > max_strain_component:
            self.best_value = max_strain_component
            context.best_params = best_params

        return penalty