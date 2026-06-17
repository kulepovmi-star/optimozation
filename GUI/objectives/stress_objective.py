from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QTabWidget, QLabel, QCheckBox
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

        self.stability = QCheckBox("Учитывать потерю устойчивости")
        layout.addWidget(self.stability)

        self.stability_ratio = QtWidgets.QDoubleSpinBox()
        layout.addWidget(QtWidgets.QLabel("Stability ratio"))
        self.stability_ratio.setRange(1, 100)
        self.stability_ratio.setValue(1.5)
        self.stability_ratio.setSingleStep(0.1)
        self.stability_ratio.setEnabled(False)
        layout.addWidget(self.stability_ratio)
        self.stability.toggled.connect(
            self.stability_ratio.setEnabled
        )
        layout.addStretch()

    def get_setting(self):
        return self.check_setting()

    def check_setting(self):
        settings={"mass": self.max_mass.value(), "displacement": self.max_strain.value(),
         "penalty": self.penalty_rate.value(), "stock_ratio_buckling":self.stability_ratio.value()}
        for k, v in settings.items():
            if v<=0:
                settings.update({k:float("inf")})
        return settings

    def get_buckling(self):
        return {"buckling":self.stability.isChecked()}