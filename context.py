from parameter_range import ParameterRangeGenerator
from creationscript import ScriptProcessor
from runner import FidesysRunner
from OptimizationMethod import OptimizationMethod
from typing import Dict, Tuple, List, TYPE_CHECKING
if TYPE_CHECKING:
    from ObjectiveFunction import OptimizationFunction


class OptimizationContext:
    def __init__(self, params, method, runner, objective, script_processor, range_params, base_dir,constraints=None):
        self.params:Dict[str, List[float]] = params
        self.objective:"OptimizationFunction" = objective  # цель расчета
        self.range_params:ParameterRangeGenerator = range_params
        self.constraints:Dict[str, float] = constraints  # что нельзя нарушать
        self.script_processor:ScriptProcessor = script_processor
        self.runner:FidesysRunner = runner
        self.method:OptimizationMethod = method
        self.base_dir:str = base_dir
        self.best_params = None


    def run_optimization(self, progress_queue):

        return self.method.optimize(self, progress_queue)



"""range_of_values=self.range_params.creating_a_range()
calculation_script=self.method.optimize(range_of_values, )
result_of_calculation=self.runner.calculation(calculation_script)
analysis=self.objective.evaluate()"""



