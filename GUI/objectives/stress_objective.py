from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QTabWidget, QLabel
from PySide6 import QtGui, QtCore, QtWidgets
class StressObjectiveWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        layout=QVBoxLayout(self)
        self.max_mass = QtWidgets.QDoubleSpinBox()
        layout.addWidget(QLabel("Mass"))
        self.max_mass.setRange(0, 10e20)
        self.max_mass.setSingleStep(100)
        layout.addWidget(self.max_mass)

        self.max_strain = QtWidgets.QDoubleSpinBox()
        layout.addWidget(QLabel("Displacement"))
        self.max_strain.setRange(0, 10e4)
        self.max_strain.setSingleStep(0.001)
        self.max_strain.setDecimals(3)
        layout.addWidget(self.max_strain)

        self.penalty_rate = QtWidgets.QSpinBox()
        layout.addWidget(QtWidgets.QLabel("Penalty ratio"))
        self.penalty_rate.setRange(10, 100)
        self.penalty_rate.setValue(20)
        self.penalty_rate.setSingleStep(1)
        layout.addWidget(self.penalty_rate)
        layout.addStretch()

    def get_setting(self):
        return self.check_setting()

    def check_setting(self):
        settings = {"mass": self.max_mass.value(), "displacement":self.max_strain.value(), "penalty":self.penalty_rate.value()}
        for k, v in settings.items():
            if v <= 0:
                settings.update({k: float("inf")})
        return settings
