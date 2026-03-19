from PySide6 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QPushButton, QVBoxLayout, QFrame
from collections import defaultdict

import math

from context import OptimizationContext
from creationscript import ScriptProcessor
from runner import FidesysRunner
from ObjectiveFunction import Mass, Strain, Stress
from OptimizationMethod import GradientDescent, BestProbe
from parameter_range import ParameterRangeGenerator
import numpy as np

"""reader=JouReader(path)
script,params=reader.read()
#optimization=OptimizationContext( params, objectives={"D":[0.3, 0.5], "M":[0.05, 0.2]}, constraints={"Stress":3.5e07, "Displacement":0.0700},)

processor = ScriptProcessor(script, params)

runner = FidesysRunner(fidesys, base_dir)

objective = Mass()

method = GradientDescent(iterations=40)

#params_range=ParameterRangeGenerator({"D":(0.3, 0.5), "M":(0.05, 0.2)}, 10)
#params_range = ParameterRangeGenerator({"r": (0.1, 0.9)},  10)
params_range = ParameterRangeGenerator({"b": (0.1, 0.2), "a": (0.3, 0.7)}, 10)


)

fc.init_application(prep_path)  # !!!!!Инициализация для версий 5.1+ (для 5.0 и ниже заменить на fc.initApplication(prep_path))
fc.start_up_no_args()  # Запуск обязательного компонента Фидесис fc"""


class TableParams(QtWidgets.QTableWidget):
    def __init__(self, parent=None):
        QtWidgets.QTableWidget.__init__(self, parent)
        self.params = None
        self.setColumnCount(3)
        header_labels = ["Params", "min", "max"]
        self.setHorizontalHeaderLabels(header_labels)

    def params_on_table(self, params):
        self.params = params
        self.setRowCount(len(params))
        for row, value in enumerate(params):
            item = QtWidgets.QTableWidgetItem(value)
            self.setItem(row, 0, item)  # только колонка 0

    def get_data(self):
        data = {}

        for row in range(self.rowCount()):
            data[self.item(row, 0).text()] = []
            for column in range(1, self.columnCount()):
                data[self.item(row, 0).text()].append(self.item(row, column).text())
        return data

    def get_params(self):
        return self.params


class TableParamsWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.vbox = QtWidgets.QVBoxLayout()
        self.label_table = QtWidgets.QLabel("Установите диапазон параметров")
        self.table = TableParams()

        # Настройка таблицы
        self.table.horizontalHeader().setStretchLastSection(True)  # Растягивать последнюю колонку
        self.table.setAlternatingRowColors(True)  # Чередование цветов строк

        # Устанавливаем ширину колонок
        self.table.setColumnWidth(0, 150)  # Первая колонка шире

        #  КНОПКИ УПРАВЛЕНИЯ ТАБЛИЦЕЙ
        button_layout = QtWidgets.QHBoxLayout()
        self.add_row_btn = QtWidgets.QPushButton("Отчистить")
        self.save_btn = QtWidgets.QPushButton("Сохранить")
        self.save_btn.clicked.connect(self.save_data)
        button_layout.addWidget(self.add_row_btn)
        button_layout.addWidget(self.save_btn)
        self.vbox.addWidget(self.label_table)
        self.vbox.addWidget(self.table)
        self.vbox.addLayout(button_layout)

        self.setLayout(self.vbox)

    def save_data(self):
        new_dict = defaultdict(list)
        data = self.table.get_data()
        for key, value in data.items():
            for i in value:
                value = float(i)
                new_dict[key].append(value)
        return new_dict


class Dialog(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(400, 600)  # Ширина 400, высота 300
        self.combo1 = QtWidgets.QComboBox()
        self.combo1.addItems(["Метод наилучшей пробы", "Градиентный спуск", "Байесовская оптимизация"])
        self.combo2 = QtWidgets.QComboBox()
        self.combo2.addItems(["Оптимизация массы", "Увеличение прочности", "Повышение жесткости"])

        self.TableParamsWidget = TableParamsWidget(self)
        self.TableParamsWidget.vbox.setContentsMargins(0, 0, 0, 0)
        self.button = QtWidgets.QPushButton("&Начать оптимизацию")
        self.label1 = QtWidgets.QLabel("Выберете метод оптимизации")
        self.label2 = QtWidgets.QLabel("Выберете цель оптимизации")
        mainBox = QtWidgets.QVBoxLayout()
        mainBox.addWidget(self.label1)
        mainBox.addWidget(self.combo1)
        mainBox.addWidget(self.label2)
        mainBox.addWidget(self.combo2)
        mainBox.addWidget(self.TableParamsWidget)
        mainBox.addWidget(self.button)
        self.setLayout(mainBox)
        self.button.clicked.connect(self.on_clicked)

    def on_clicked(self):
        data = self.TableParamsWidget.save_data()
        params = self.TableParamsWidget.get_params()
        method = self.combo1.currentText()
        print(method)
        task = self.combo2.currentText()
        print(task)
        print(data)

        optimization = OptimizationContext(params, objectives={"D": [0.3, 0.5], "M": [0.05, 0.2]},
                                           constraints={"Stress": 3.5e07, "Displacement": 0.0700}, )

        processor = ScriptProcessor(script, params)

        runner = FidesysRunner(fidesys, base_dir)

        objective = Mass()

        method = self.get_method()(40)

        # params_range=ParameterRangeGenerator({"D":(0.3, 0.5), "M":(0.05, 0.2)}, 10)
        # params_range = ParameterRangeGenerator({"r": (0.1, 0.9)},  10)
        params_range = ParameterRangeGenerator({"b": (0.1, 0.2), "a": (0.3, 0.7)}, 10)
        context = OptimizationContext(
            params=params,
            script_processor=processor,
            runner=runner,
            objective=objective,
            method=method,
            range_params=params_range,
            constraints={"Stress": 4e08, "Displacement": 0.300, "Mass": 0},
            base_dir=base_dir)

    def set_params(self, params):
        return self.TableParamsWidget.table.params_on_table(params)

    def add_button(self):
        button = QPushButton('Кнопка')
        self.layout_buttons.insertWidget(0, button)  # Добавление в начало, с пружиной это прижмет вверх
        # self.layout_buttons.addWidget(button)  # Добавление в конец, с пружиной это прижмет вниз

    def aprepro_params(self):
        pass

    def get_method(self):
        methods = {"Метод наилучшей пробы": BestProbe,
                   "Градиентный спуск": GradientDescent,
                   "Байесовская оптимизация": None}
        return methods[self.combo1.currentText()]


