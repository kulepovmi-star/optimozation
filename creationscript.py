import math
import re
from typing import List, Dict

functions={
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'sqrt': math.sqrt,
            'exp': math.exp,
            'log': math.log,
            'ln': math.log,
            'abs': abs,
            'min': min,
            'max': max,
        }


def _evaluate(expression: str, params:Dict) -> str:
    # вычисления
    namespace = functions.copy()
    namespace.update(params)
    if expression in params:
        return str(params[expression])
    else:
        result = eval(expression, {"__builtins__": {}}, namespace)
        return str(result)


class ScriptProcessor:
    def __init__(self, template_script: List[str], base_params: dict):
        self.template_script = template_script
        self.base_params = base_params

    def _process_lines(self, params) -> List[str]:
        # разбор скрипта
        result_script = []
        pattern = re.compile(r'\{([^}]+)}')
        for line in self.template_script:
            processed = line
            expressions = re.findall(pattern, line)
            if expressions:
                for expression in expressions:
                    calculated = _evaluate(expression, params)
                    processed = processed.replace('{' + expression + '}', calculated)
            result_script.append(processed)
        return result_script

    def build(self, params: dict) -> List[str]:
        merged = {**self.base_params, **params}
        return self._process_lines(merged)

