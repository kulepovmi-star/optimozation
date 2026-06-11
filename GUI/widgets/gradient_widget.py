from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QTabWidget, QLabel
from PySide6 import QtGui, QtCore, QtWidgets


class GradientSettingsWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.setup_defaults()

    def setup_ui(self):
        layout=QVBoxLayout(self)
        self.iterations_spin = QtWidgets.QSpinBox()
        self.learning_rate = QtWidgets.QDoubleSpinBox()
        self.precision_spin = QtWidgets.QDoubleSpinBox()
        self.b1 = QtWidgets.QDoubleSpinBox()
        self.b2 = QtWidgets.QDoubleSpinBox()

        layout.addWidget(QLabel("Iteration"))
        layout.addWidget(self.iterations_spin)

        layout.addWidget(QLabel("Steps"))
        layout.addWidget(self.precision_spin)

        layout.addWidget(QLabel("B1"))
        layout.addWidget(self.b1)

        layout.addWidget(QLabel("B2"))
        layout.addWidget(self.b2)

        layout.addWidget(QLabel("Learning rate"))
        layout.addWidget(self.learning_rate)
        layout.addStretch()

    def get_setting(self):
        return {"iterations": self.iterations_spin.value(), "learning_rate": self.learning_rate.value(),
                "b1":self.b1.value(), "b2":self.b2.value(), "steps":self.iterations_spin.value()}

    def setup_defaults(self):
        self.iterations_spin.setRange(1, 10000)
        self.iterations_spin.setValue(40)

        self.precision_spin.setRange(0.001, 0.1)
        self.precision_spin.setDecimals(4)
        self.precision_spin.setValue(0.01)
        self.precision_spin.setSingleStep(0.01)

        self.learning_rate.setRange(0.001, 0.1)
        self.learning_rate.setDecimals(4)
        self.learning_rate.setValue(0.04)
        self.learning_rate.setSingleStep(0.01)

        self.b1.setRange(0, 0.9)
        self.b1.setDecimals(2)
        self.b1.setValue(0.7)
        self.b1.setSingleStep(0.1)

        self.b2.setRange(0, 0.99)
        self.b2.setDecimals(2)
        self.b2.setValue(0.90)
        self.b2.setSingleStep(0.1)



