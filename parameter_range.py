from collections import defaultdict

import numpy as np
from typing import Dict, Tuple, List, TypedDict
from functools import singledispatchmethod

class ParameterRangeGenerator:
    def __init__(self, params: Dict [str, List[float]], steps:int) -> None:
        self.changeable_parameters=params.copy()
        self.steps=steps


    @singledispatchmethod
    def creating_a_range(self, arg):
        raise NotImplementedError("Неподдерживаемый тип аргумента")

    @creating_a_range.register(type(None))
    def _(self, arg: None) -> Dict[str, List[float]]:
        range_dict = {}
        if self.steps:
            for key, (min_val, max_val) in self.changeable_parameters.items():
                range_dict[key] = list(np.linspace(min_val, max_val, self.steps))
            return range_dict
        else:
            return self.changeable_parameters

    @creating_a_range.register(dict)
    def _(self, best_params: Dict[str, float]) -> Dict[str, List[float]]:
        range_dict = {}
        factor = 0.3
        for key, center in best_params.items():
            if key not in self.changeable_parameters:
                continue

            orig_min, orig_max = self.changeable_parameters[key]
            delta = abs(center * factor)

            min_val = max(center - delta, orig_min)
            max_val = min(center + delta, orig_max)

            range_dict[key] = list(np.linspace(min_val, max_val, self.steps))

        return range_dict

    def range_by_step(self, params_step):
        new_range={}
        for key, (min_val, max_val) in self.changeable_parameters.items():
            print(min_val, max_val, params_step[key])
            new_range.update({key:list(np.arange(float(min_val), float(max_val), float(params_step[key])))})
            new_range[key].append(float(max_val))
        print(new_range)
        return new_range
