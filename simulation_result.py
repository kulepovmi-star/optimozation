import re
import vtk  # Библиотека работы с выходными данными
from vtk.util.numpy_support import vtk_to_numpy  # Модуль для преобразования результато
import os

class SimulationResult:

    mass = None
    stress_list = None
    strain_list = None


    def point_data(self, base_dir:str):
        reader = vtk.vtkXMLUnstructuredGridReader()  # Подключаем читалку
        print("Читаем результаты из ",
              str(base_dir) + r"\1\case1_step0001_substep0001.vtu")  # Пишет откуда берем результаты
        filename = os.path.join(str(base_dir) + r"\1\case1_step0001_substep0001.vtu")  # Указываем путь к файлу
        mass_file = os.path.join(str(base_dir) + r"\1\PreciseMassSummary.log")

        reader.SetFileName(filename)  # Подключаем путь к читалке и читаем
        reader.Update()  # Needed because of GetScalarRange
        grid = reader.GetOutput()  # Забираем выходные данные
        point_data = grid.GetPointData()  # Забираем данные для точек
        mass=self._read_mass(mass_file)
        return point_data, mass

    def _read_mass(self, mass_file: str) -> float:
        with open(mass_file) as f:
            for line in f:
                match = re.search(r'TOTAL MASS\s*=\s*(\d+\.?\d*(?:[eE][+-]?\d+)?)', line)
                if match:
                    mass = float(match.group(1))
                    break
        return mass

    def save_data(self, base_dir:str):
        point_data, self.mass=self.point_data(base_dir)
        self.stress_list = vtk_to_numpy(point_data.GetArray("Stress"))  # Считываем напряжения из массива результатов
        self.strain_list = vtk_to_numpy(point_data.GetArray("Displacement"))
        return self







