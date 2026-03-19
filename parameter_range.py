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
        for key, (min_val, max_val) in self.changeable_parameters.items():
            range_dict[key] = list(np.linspace(min_val, max_val, self.steps))
        return range_dict

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