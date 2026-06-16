from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QTabWidget, QLabel
from PySide6 import QtGui, QtCore, QtWidgets

class ProbeSettingsWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.setup_defaults()

    def setup_ui(self):
        layout=QVBoxLayout(self)
        self.iterations_spin = QtWidgets.QSpinBox()
        layout.addWidget(QLabel("Iteration"))
        layout.addWidget(self.iterations_spin)
        self.grid = QtWidgets.QSpinBox()
        layout.addWidget(QLabel("Grid"))
        layout.addWidget(self.grid)
        layout.addStretch()

    def get_setting(self):
        return {"iterations": self.iterations_spin.value()}

    def get_grid(self):
        return self.grid.value()

    def setup_defaults(self):
        self.iterations_spin.setRange(1, 10000)
        self.iterations_spin.setValue(40)
        self.grid.setRange(1, 101)
        self.grid.setValue(100)

