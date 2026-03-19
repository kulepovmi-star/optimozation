import re
from typing import Tuple, List, Dict
import numpy as np

class JouReader:

    def __init__(self,path):
        self.path=path
    def read(self) -> Tuple[List[str], Dict[str, List[float]]]:
        aprepro_params = {}
        script = []
        with open(self.path,'r', encoding="UTF-8") as f:
            for line in f:
                if line.strip():
                    match=re.search(r'#\{(\w+)\s*=\s*([^}]+)}', line)
                    if match:
                        aprepro_params[match.group(1)] = float(match.group(2))
                    else:
                        """reset=re.search(r'undo', line)
                        if "undo" == line.strip():
                            script.pop()
                        else:"""
                        script.append(line.strip())
        #print(*script, sep='\n')
        return script, aprepro_params



