from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QTabWidget, QLabel
from PySide6 import QtGui, QtCore, QtWidgets
class StrainObjectiveWidget(QWidget):
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

        self.max_stress = QtWidgets.QDoubleSpinBox()
        layout.addWidget(QLabel("Stress"))
        self.max_stress.setRange(0, 10e20)
        self.max_stress.setSingleStep(100000)
        layout.addWidget(self.max_stress)

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
        settings={"mass": self.max_mass.value(), "stress":self.max_stress.value(), "penalty":self.penalty_rate.value()}

        for k, v in settings.items():
            if v<=0:
                settings.update({k:float("inf")})
        return settings
