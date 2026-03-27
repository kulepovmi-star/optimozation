from PySide6 import QtGui, QtCore, QtWidgets
from PySide6.QtWidgets import QPushButton, QMessageBox
from collections import defaultdict
import os
from multiprocessing import Process, Queue
from worker import optimization_process
from context import OptimizationContext
from creationscript import ScriptProcessor
from runner import FidesysRunner
from ObjectiveFunction import Mass, Strain, Stress
from OptimizationMethod import GradientDescent, BestProbe, Bayesian_optimization
from parameter_range import ParameterRangeGenerator
import numpy as np

base_dir = os.path.dirname(os.path.abspath(__file__))  # Директория где лежит скрипт

class OptimizationWorker(QtCore.QObject):
    finished = QtCore.Signal()

    def __init__(self, context):
        super().__init__()
        self.context = context
        self._running = True

    @QtCore.Slot()
    def run(self):
        try:
            self.context.run_optimization()
        finally:
            self.finished.emit()

    def stop(self):
        self._running = False

class TableParams(QtWidgets.QTableWidget):
    def __init__(self, parent=None):
        QtWidgets.QTableWidget.__init__(self, parent)
        self.params = None
        self.setColumnCount(3)
        header_labels = ["Params", "min", "max"]
        self.setHorizontalHeaderLabels(header_labels)

    def params_on_table(self, params):
        self.params = params
        self.setRowCount(len(self.params))
        for row, value in enumerate(self.params):
            item = QtWidgets.QTableWidgetItem(value)
            self.setItem(row, 0, item)  # только колонка 0

    def del_item(self):

        for row, _ in enumerate(self.params):
            for i in [1, 2]:
                self.takeItem(row, i)

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
        self.table.setColumnWidth(0, 130)  # Первая колонка шире

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
        self.add_row_btn.clicked.connect(self.clean)

    def save_data(self):
        new_dict = defaultdict(list)
        data = self.table.get_data()
        for key, value in data.items():
            for i in value:
                value = float(i)
                new_dict[key].append(value)
        return new_dict

    def clean(self):
        self.table.del_item()


class Dialog(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.script = None
        self.setFixedSize(400, 600)  # Ширина 400, высота 300

        self.TableParamsWidget = TableParamsWidget(self)
        self.combo1 = QtWidgets.QComboBox()
        self.combo2 = QtWidgets.QComboBox()

        self.button = QtWidgets.QPushButton("&Начать оптимизацию")
        self.label1 = QtWidgets.QLabel("Выберете метод оптимизации")
        self.label2 = QtWidgets.QLabel("Выберете цель оптимизации")

        self.tabs = QtWidgets.QTabWidget()

        # страницы
        self.main_tab = QtWidgets.QWidget()
        self.main_task= QtWidgets.QWidget()
        self.advanced_tab = QtWidgets.QWidget()

        # добавляем вкладки
        self.tabs.addTab(self.main_tab, "Главная")
        self.tabs.addTab(self.main_task, "Целевая функция")
        self.tabs.addTab(self.advanced_tab, "Расширенные настройки")

        # заполняем вкладки
        self.setup_main_tab()
        self.setup_main_task()
        self.setup_advanced_tab()

        # главный layout окна
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.tabs)

    def setup_main_tab(self):
        layout = QtWidgets.QVBoxLayout(self.main_tab)

        self.TableParamsWidget.vbox.setContentsMargins(0, 0, 0, 0)

        self.combo1.addItems([
            "Метод наилучшей пробы",
            "Градиентный спуск",
            "Байесовская оптимизация"
        ])

        self.combo2.addItems([
            "Оптимизация массы",
            "Увеличение прочности",
            "Повышение жесткости"
        ])
        self.button.clicked.connect(self.on_clicked)
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        layout.addWidget(self.label1)
        layout.addWidget(self.combo1)
        layout.addWidget(self.label2)
        layout.addWidget(self.combo2)
        layout.addWidget(self.TableParamsWidget)
        layout.addWidget(self.button)
        layout.addWidget(self.progress_bar)
        self.main_tab.setLayout(layout)

    def setup_advanced_tab(self):

        layout = QtWidgets.QVBoxLayout(self.advanced_tab)
        self.method_stack = QtWidgets.QStackedWidget()

        # страницы
        self.method_stack.addWidget(self.widget_best_probe())
        self.method_stack.addWidget(self.widget_gradient())
        self.method_stack.addWidget(self.widget_bayesian())

        layout.addWidget(self.method_stack)

        self.advanced_tab.setLayout(layout)

        # связь с combo
        self.combo1.currentIndexChanged.connect(
            self.method_stack.setCurrentIndex
        )

    def setup_main_task(self):

        layout = QtWidgets.QVBoxLayout(self.main_task)
        self.method_stack_task = QtWidgets.QStackedWidget()

        # страницы
        self.method_stack_task.addWidget(self.task_mass())
        self.method_stack_task.addWidget(self.task_stress())
        self.method_stack_task.addWidget(self.task_strain())

        layout.addWidget(self.method_stack_task)
        self.main_task.setLayout(layout)

        # связь с combo
        self.combo2.currentIndexChanged.connect(
            self.method_stack_task.setCurrentIndex
        )

    def get_method_params(self, current_widget):
        params = {}

        # целые числа
        for child in current_widget.findChildren(QtWidgets.QSpinBox):
            params[child.objectName()] = child.value()

        # числа с плавающей точкой
        for child in current_widget.findChildren(QtWidgets.QDoubleSpinBox):
            params[child.objectName()] = child.value()

        return params

    def widget_best_probe(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        label = QtWidgets.QLabel("Метод наилучшей пробы")
        layout.addWidget(label)
        self.iterations_spin = QtWidgets.QSpinBox()
        self.iterations_spin.setRange(1, 10000)
        self.iterations_spin.setValue(40)
        self.iterations_spin.setObjectName("iterations")

        layout.addWidget(QtWidgets.QLabel("Количество итераций"))
        layout.addWidget(self.iterations_spin)

        layout.addStretch()
        return widget

    def widget_gradient(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        label = QtWidgets.QLabel("Градиентный спуск")
        layout.addWidget(label)
        self.iterations_spin = QtWidgets.QSpinBox()
        self.iterations_spin.setRange(1, 10000)
        self.iterations_spin.setValue(40)
        self.iterations_spin.setObjectName("iterations")

        self.precision_spin = QtWidgets.QDoubleSpinBox()
        self.precision_spin.setRange(0.001, 0.1)
        self.precision_spin.setDecimals(4)
        self.precision_spin.setValue(0.01)
        self.precision_spin.setSingleStep(0.01)
        self.precision_spin.setObjectName("steps")

        self.learning_rate = QtWidgets.QDoubleSpinBox()
        self.learning_rate.setRange(0.001, 0.1)
        self.learning_rate.setDecimals(4)
        self.learning_rate.setValue(0.04)
        self.learning_rate.setSingleStep(0.01)
        self.learning_rate.setObjectName("l_r")

        self.b1 = QtWidgets.QDoubleSpinBox()
        self.b1.setRange(0, 0.9)
        self.b1.setDecimals(2)
        self.b1.setValue(0.05)
        self.b1.setSingleStep(0.1)
        self.b1.setObjectName("b1")

        self.b2 = QtWidgets.QDoubleSpinBox()
        self.b2.setRange(0, 0.99)
        self.b2.setDecimals(2)
        self.b2.setValue(0.80)
        self.b2.setSingleStep(0.1)
        self.b2.setObjectName("b2")

        layout.addWidget(QtWidgets.QLabel("Количество итераций"))
        layout.addWidget(self.iterations_spin)

        layout.addWidget(QtWidgets.QLabel("Дельта"))
        layout.addWidget(self.precision_spin)
        layout.addWidget(QtWidgets.QLabel("learning rate"))
        layout.addWidget(self.learning_rate)

        layout.addWidget(QtWidgets.QLabel("b1"))
        layout.addWidget(self.b1)
        layout.addWidget(QtWidgets.QLabel("b2"))
        layout.addWidget(self.b2)
        layout.addStretch()
        return widget

    def widget_bayesian(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        label = QtWidgets.QLabel("Байесовская оптимизация")
        layout.addWidget(label)

        layout.addStretch()
        return widget



    def task_mass(self):

        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        label = QtWidgets.QLabel("Масса")
        layout.addWidget(label)

        self.max_stress = QtWidgets.QDoubleSpinBox()
        self.max_stress.setObjectName("Stress")
        self.max_stress.setRange(0, 10e20)

        self.max_strain = QtWidgets.QDoubleSpinBox()
        self.max_strain.setObjectName("Displacement")
        self.max_strain.setRange(0, 10e3)

        layout.addWidget(QtWidgets.QLabel("Предельные напряжения"))
        layout.addWidget(self.max_stress)
        layout.addWidget(QtWidgets.QLabel("Максимальные перемещения"))
        layout.addWidget(self.max_strain)

        layout.addStretch()
        return widget

    def task_stress(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        label = QtWidgets.QLabel("Прочность")
        layout.addWidget(label)
        self.max_mass = QtWidgets.QDoubleSpinBox()
        self.max_mass.setObjectName("Mass")
        self.max_mass.setRange(0, 10e7)

        self.max_strain = QtWidgets.QDoubleSpinBox()
        self.max_strain.setObjectName("Displacement")
        self.max_strain.setRange(0, 10e3)

        layout.addWidget(QtWidgets.QLabel("Максимальная масса"))
        layout.addWidget(self.max_mass)
        layout.addWidget(QtWidgets.QLabel("Максимальные перемещения"))
        layout.addWidget(self.max_strain)
        layout.addStretch()
        return widget

    def task_strain(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        label = QtWidgets.QLabel("Жесткость")
        layout.addWidget(label)
        self.max_mass = QtWidgets.QDoubleSpinBox()
        self.max_mass.setObjectName("Mass")
        self.max_mass.setRange(0, 10e7)


        self.max_stress = QtWidgets.QDoubleSpinBox()
        self.max_stress.setObjectName("Stress")
        self.max_stress.setRange(0, 10e20)

        layout.addWidget(QtWidgets.QLabel("Максимальная масса"))
        layout.addWidget(self.max_mass)

        layout.addWidget(QtWidgets.QLabel("Предельные напряжения"))
        layout.addWidget(self.max_stress)

        layout.addStretch()
        return widget


    def on_clicked(self):
        current_widget_settings = self.method_stack.currentWidget()
        current_widget_tasks = self.method_stack_task.currentWidget()

        data = {
            "script": self.script,
            "params": self.TableParamsWidget.table.get_params(),
            "ranges": self.TableParamsWidget.save_data(),
            "method": self.combo1.currentText(),
            "objective": self.combo2.currentText(),
            "method_params": self.get_method_params(current_widget_settings),
            "constraints": {
                **self.get_method_params(current_widget_tasks)
            },
            "base_dir": base_dir
        }

        self.queue = Queue()

        self.process = Process(
            target=optimization_process,
            args=(data, self.queue)
        )

        self.process.start()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.check_queue)
        self.timer.start(100)

    def check_queue(self):

        while not self.queue.empty():

            msg, value = self.queue.get()

            if msg == "progress":
                self.progress_bar.setValue(value)

            elif msg == "finished":
                self.timer.stop()
                self.progress_bar.setValue(100)
                print("Оптимизация завершена")

    def stop_all_tasks(self):
        print("Stopping optimization...")


    def set_params(self, params):
        return self.TableParamsWidget.table.params_on_table(params)

    def set_script(self, script):
        self.script=script

    def add_button(self):
        button = QPushButton('Кнопка')
        self.layout_buttons.insertWidget(0, button)  # Добавление в начало, с пружиной это прижмет вверх


    def closeEvent(self, event):
        reply = QMessageBox.question(
            self,
            'Подтверждение закрытия',
            'Вы действительно хотите закрыть приложение?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            try:
                if self.process and self.process.is_alive():
                    self.process.terminate()
                    self.process.join(timeout=2)
                    print("Оптимизация остановлена")
            except Exception as e:
                print("Ошибка завершения:", e)

            event.accept()
        else:
            event.ignore()

    def get_method(self):
        methods = {"Метод наилучшей пробы": BestProbe,
                   "Градиентный спуск": GradientDescent,
                   "Байесовская оптимизация": Bayesian_optimization}
        return methods[self.combo1.currentText()]

    def get_task(self):
        tasks = {"Оптимизация массы":Mass, "Увеличение прочности":Stress, "Повышение жесткости":Strain}
        return tasks[self.combo2.currentText()]


