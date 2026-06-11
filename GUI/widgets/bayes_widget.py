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
        layout.addStretch()

    def get_setting(self):
        return {"iterations": self.iterations_spin.value()}

    def setup_defaults(self):
        self.iterations_spin.setRange(1, 10000)
        self.iterations_spin.setValue(40)
