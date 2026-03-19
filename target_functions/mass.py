import time

from vtk.util.numpy_support import vtk_to_numpy  # Модуль для преобразования результатов
import re



STRESS_TOLERANCE = 1  # Запас прочности (почему-то результаты завышены на 10%)
def save():
    base_param = float("inf")
    def check(aprepro, mass, script_result, aprepro_new_params):
        nonlocal base_param
        if base_param > mass:
            base_param = mass
            aprepro.best_script = script_result
            aprepro.best_mass = mass
            aprepro.best_params = aprepro_new_params
        return mass
    return check

save_data=save()

def read_mass(mass_data):
    with open(mass_data) as f:
        for line in f:
            match = re.search(r'TOTAL MASS\s*=\s*(\d+\.?\d*(?:[eE][+-]?\d+)?)', line)
            if match:
                mass = float(match.group(1))
                break
    return mass

def mass(point_data, mass_data, aprepro, result_script, aprepro_new_params):
    mass = read_mass(mass_data)
    if mass:
        arrayOfStress =vtk_to_numpy(point_data.GetArray("Stress"))  # Считываем напряжения из массива результатов
        arrayOfDisplacement =vtk_to_numpy(point_data.GetArray("Displacement"))
        for point in range(len(arrayOfStress)): #
            if  arrayOfStress[point][6] > aprepro.constraints.get("Stress", 0)*STRESS_TOLERANCE or max(arrayOfDisplacement[point])>aprepro.constraints.get("Displacement", 0):  # Проверяем напряжения по Мизесу в узлах и максимальные перемещения
                print(arrayOfStress[point][6], "напряжения", arrayOfDisplacement[point], "перемещения")
                return None
        else:
            return save_data(aprepro, mass, result_script, aprepro_new_params)
    else:
        return None

