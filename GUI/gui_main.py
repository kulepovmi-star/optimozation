from PySide6.QtWidgets import QMainWindow, QWidget, QMessageBox, QVBoxLayout, QTabWidget, QComboBox, QStackedWidget, QLabel
from PySide6 import QtWidgets, QtCore
import os
from multiprocessing import Process, Queue
from GUI.widgets.gradient_widget import GradientSettingsWidget
from GUI.widgets.best_probe_widget import ProbeSettingsWidget
from GUI.widgets.bayes_widget import BayesSettingsWidget
from GUI.widgets.step_be_step_widget import StepSettingsWidget
from GUI.objectives.mass_objective import MassObjectiveWidget
from GUI.objectives.strain_objective import StrainObjectiveWidget
from GUI.objectives.stress_objective import StressObjectiveWidget
from GUI.table_params import TableParamsWidget
from worker import optimization_process

class MainWindow(QMainWindow):

    def __init__(self,params,script, base_dir):
        super().__init__()
        self.params = params
        self.script=script
        self.setup_ui()
        self.set_params()
        self.process = None
        self.base_dir = base_dir # Директория где лежит скрипт


    def setup_ui(self):
        self.resize(450, 570)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        # main layout
        self.layout = QVBoxLayout(self.central_widget)
        self.tabs = QTabWidget()
        self.main_tab = QWidget()
        self.objective_tab = QWidget()
        self.advanced_tab = QWidget()
        self.tabs.addTab(self.main_tab, "Main")
        self.tabs.addTab(self.objective_tab, "Objective")
        self.tabs.addTab(self.advanced_tab, "Advanced")
        self.layout.addWidget(self.tabs)
        self.setup_main_tab()
        self.setup_objective_tab()
        self.setup_advanced_tab()


    def setup_main_tab(self):
        layout=QVBoxLayout(self.main_tab)
        layout.addWidget(QLabel("Выберете метод оптимизации"))
        self.method_combo=QComboBox()
        self.method_combo.addItems([
            "Best Probe",
            "Gradient method",
            "Bayesian",
            "Step by step"
        ])
        self.method_stack = QStackedWidget()
        self.method_combo.currentIndexChanged.connect(
            self.method_stack.setCurrentIndex)

        self.method_combo.currentIndexChanged.connect(
            self.on_method_changed)
        layout.addWidget(self.method_combo)
        layout.addWidget(QLabel("Выберете цель оптимизации"))

        self.purpose_combo = QComboBox()
        self.purpose_combo.addItems([
            "Mass",
            "Stress",
            "Strain",
        ])
        self.purpose_stack = QStackedWidget()
        self.purpose_combo.currentIndexChanged.connect(
            self.purpose_stack.setCurrentIndex)
        layout.addWidget(self.purpose_combo)

        self.step_step = StepSettingsWidget(self.params)
        self.main_table = TableParamsWidget()

        layout.addWidget(self.step_step)
        self.step_step.hide()
        layout.addWidget(self.main_table)
        self.button = QtWidgets.QPushButton("&Start")
        self.button.clicked.connect(self.start)
        self.stop_button = QtWidgets.QPushButton("&Stop")
        self.stop_button.clicked.connect(self.stop)
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.button)
        layout.addLayout(button_layout)
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        layout.addStretch()

    def on_method_changed(self, index):
        is_step = index == 3

        self.step_step.setVisible(is_step)
        self.main_table.setVisible(not is_step)

    def setup_objective_tab(self):
        layout = QVBoxLayout(self.objective_tab)
        layout.addWidget(QLabel("Настройки целевой функции"))
        self.mass = MassObjectiveWidget()
        self.stress=StressObjectiveWidget()
        self.strain=StrainObjectiveWidget()
        self.purpose_stack.addWidget(self.mass)
        self.purpose_stack.addWidget(self.stress)
        self.purpose_stack.addWidget(self.strain)
        self.button = QtWidgets.QPushButton("&Начать оптимизацию")
        self.button.clicked.connect(self.start)
        layout.addWidget(self.purpose_stack)
        layout.addStretch()

    def setup_advanced_tab(self):
        layout = QVBoxLayout(self.advanced_tab)
        self.gradient = GradientSettingsWidget()
        self.best_probe= ProbeSettingsWidget()
        self.bayes=BayesSettingsWidget()

        self.method_stack.addWidget(self.best_probe)
        self.method_stack.addWidget(self.gradient)
        self.method_stack.addWidget(self.bayes)
        self.method_stack.addWidget(self.step_step.empty_window)
        layout.addWidget(self.method_stack)
        layout.addStretch()

    def start(self):

        method = self.method_stack.currentWidget()
        objective = self.purpose_stack.currentWidget()
        table=self.get_current_table()
        if table.check_data():
            self.progress_bar.setValue(1)
            data = {
                "script": self.script,
                "params": self.main_table.table.get_params(),
                "ranges": table.save_data(),
                "method": self.method_combo.currentText(),
                "objective": self.purpose_combo.currentText(),
                "method_params": {**method.get_setting()},
                "constraints": {
                    **objective.get_setting(), **objective.get_buckling()
                },
                "base_dir": self.base_dir,
                "grid":method.get_grid()
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
        else:
            QMessageBox.information(
                self,
                "Информация",
                "Введите корректные данные"
            )

    def stop(self):
        reply = QMessageBox.question(
            self,
            'Подтверждение остановки',
            'Вы действительно хотите остановить процесс?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            try:
                self.progress_bar.setValue(0)
                if self.process and self.process.is_alive():
                    self.process.terminate()
                    self.process.join(timeout=2)
                    print("Оптимизация остановлена")
            except Exception as e:
                print("Ошибка завершения:", e)


    def get_current_table(self):
        if self.method_combo.currentIndex() == 3:
            return self.step_step
        return self.main_table

    def set_params(self):
        self.main_table.table.params_on_table(self.params)

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

    def check_queue(self):

        while not self.queue.empty():

            msg, value = self.queue.get()

            if msg == "progress":
                self.progress_bar.setValue(value)

            elif msg == "finished":
                self.timer.stop()
                self.progress_bar.setValue(100)
                print("Оптимизация завершена")
