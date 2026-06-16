from fidesys_env import setup_fidesys
from context import OptimizationContext
from creationscript import ScriptProcessor
from runner import FidesysRunner
from parameter_range import ParameterRangeGenerator
from ObjectiveFunction import Mass, Strain, Stress
from OptimizationMethod import GradientDescent, BestProbe, Bayesian_optimization, Step_by_step_change
import time

def optimization_process(data, progress_queue):
    cubit, fidesys, fc = setup_fidesys()


    processor = ScriptProcessor(
        data["script"],
        data["params"]
    )

    runner = FidesysRunner(data["base_dir"],fidesys)
    objective_cls = {
        "Mass":Mass,
        "Stress":Stress,
        "Strain":Strain
    }[data["objective"]]

    method_cls = {
        "Best Probe": BestProbe,
        "Gradient method": GradientDescent,
        "Bayesian": Bayesian_optimization,
        "Step by step":Step_by_step_change
    }[data["method"]]

    method = method_cls(**data["method_params"])


    params_range = ParameterRangeGenerator(
        data["ranges"],
        data["grid"]
    )

    context = OptimizationContext(
        params=data["params"],
        script_processor=processor,
        runner=runner,
        objective=objective_cls(),
        method=method,
        range_params=params_range,
        constraints=data["constraints"],
        base_dir=data["base_dir"]
    )


    # запуск
    context.run_optimization(progress_queue)