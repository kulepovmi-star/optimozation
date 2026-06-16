from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QTabWidget, QLabel
from PySide6 import QtGui, QtCore, QtWidgets

class BayesSettingsWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.setup_defaults()

    def setup_ui(self):
        layout=QVBoxLayout(self)
        self.iterations_spin = QtWidgets.QSpinBox()
        layout.addWidget(QLabel("Iteration"))
        layout.addWidget(self.iterations_spin)
        self.b = QtWidgets.QDoubleSpinBox()
        layout.addWidget(QLabel("Upper Confidence Bound betta"))
        layout.addWidget(self.b)
        self.selection = QtWidgets.QSpinBox()
        layout.addWidget(QLabel("Initial selection for each of the parameters"))
        layout.addWidget(self.selection)
        self.grid = QtWidgets.QSpinBox()
        layout.addWidget(QLabel("Grid"))
        layout.addWidget(self.grid)
        layout.addStretch()
        #UpperConfidenceBound

    def get_setting(self):
        return {"iterations": self.iterations_spin.value(), "selection": self.selection.value(),"b": self.b.value(), }

    def get_grid(self):
        return self.grid.value()

    def setup_defaults(self):
        self.iterations_spin.setRange(1, 10000)
        self.iterations_spin.setValue(40)
        self.grid.setRange(1, 1000)
        self.grid.setValue(100)
        self.selection.setRange(1, 100)
        self.selection.setValue(5)
        self.b.setRange(0.1, 100)
        self.b.setValue(2)
        self.b.setDecimals(2)
        self.b.setSingleStep(0.5)

