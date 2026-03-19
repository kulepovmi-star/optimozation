import random
import vtk  # Библиотека работы с выходными данными
from vtk.util.numpy_support import vtk_to_numpy  # Модуль для преобразования результатов
import os
import sys



class FidesysRunner:

    def __init__(self,  base_dir:str, fidesys):

        self.base_dir = base_dir
        self.fidesys= fidesys

    def calculation(self, result_script):
        output_pvd_path = os.path.join(self.base_dir + "\\" + f"1.pvd")  # Объявляем директорию и файл сохранения
        for line in result_script:
            print(line, "cmd")
            self.fidesys.cmd(line)

        self.fidesys.cmd("analysis type static elasticity dim3")
        self.fidesys.cmd("calculation start path '" + output_pvd_path + "'")