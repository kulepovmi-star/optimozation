from PySide6.QtWidgets import QMainWindow, QWidget, QMessageBox, QVBoxLayout, QTabWidget, QLabel, QCheckBox, QPushButton
from PySide6 import QtGui, QtCore, QtWidgets
from collections import defaultdict

class EmptyWindow(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        layout=QVBoxLayout(self)
        self.grid = QtWidgets.QSpinBox()
        self.grid.setValue(0)
        self.checkbox = QCheckBox("Рассматривать все комбинации")
        layout.addWidget(self.checkbox)
        layout.addStretch()

    def get_setting(self):
        return {"checkbox": self.checkbox.isChecked()}

    def get_grid(self):
        return self.grid.value()


class WidgetforTable(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.vbox = QtWidgets.QVBoxLayout()
        self.vbox.setContentsMargins(0, 0, 0, 0)
        self.label_table = QtWidgets.QLabel("Установите значения параметров")
        self.table = Table()

        # Настройка таблицы
        self.table.horizontalHeader().setStretchLastSection(True)  # Растягивать последнюю колонку
        self.table.setAlternatingRowColors(True)  # Чередование цветов строк

        # Устанавливаем ширину колонок
        self.table.setColumnWidth(0, 130)  # Первая колонка шире
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        button_layout = QtWidgets.QHBoxLayout()
        btn_add = QPushButton("добавить столбец")
        btn_del = QPushButton("удалить столбец")
        self.vbox.addWidget(self.table)
        btn_add.clicked.connect(self.add_column)
        btn_del.clicked.connect(self.del_column)
        button_layout.addWidget(btn_del)
        button_layout.addWidget(btn_add)
        self.vbox.addLayout(button_layout)
        self.setLayout(self.vbox)

    def set_params(self,params):
        self.table.params_on_table(params)

    def save_data(self):
        return self.table.get_data()

    def add_column(self):
        col = self.table.columnCount() # текущее число столбцов
        self.table.insertColumn(col)  # вставляем новый
        self.table.setHorizontalHeaderItem(
            col,
            QtWidgets.QTableWidgetItem(f"value {col}"))

    def del_column(self):
        col = self.table.columnCount()
        if col > 1:
            self.table.removeColumn(col - 1)

class Table(QtWidgets.QTableWidget):
    def __init__(self):
        super().__init__()
        self.setColumnCount(2)
        header_labels = ["Params", "value 1"]
        self.setHorizontalHeaderLabels(header_labels)
        self.data={}
        self.setMinimumHeight(300)

    def params_on_table(self, params):
        self.setRowCount(len(params))
        if self.data:
            self.setRowCount(len(self.data))
            self.setColumnCount(max(len(v) for v in self.data.values()) + 1)
            for id, (key, values) in enumerate(self.data.items()):
                item = QtWidgets.QTableWidgetItem(key)
                self.setItem(id, 0, item)  # только колонка 0
                for column, value in enumerate(values):
                    item = QtWidgets.QTableWidgetItem(str(value))
                    self.setItem(id, column + 1, item)
        else:
            for row, value in enumerate(params):
                item = QtWidgets.QTableWidgetItem(value)
                self.setItem(row, 0, item)  # только колонка 0

    def get_data(self):
        new_data=defaultdict(list)
        for row in range(self.rowCount()):
            for column in range(1, self.columnCount()):
                new_data[self.item(row, 0).text()].append(float(self.item(row, column).text()))
        self.data=new_data
        return self.data


class StepSettingsWidget(QWidget):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.setup_ui()


    def setup_ui(self):
        layout=QVBoxLayout(self)
        label = QtWidgets.QLabel("Оптимизация по параметрам")
        layout.addWidget(label)
        layout.setContentsMargins(0, 0, 0, 0)

        self.empty_window=EmptyWindow()
        self.TableParamsWidget=WidgetforTable()

        self.TableParamsWidget.set_params(self.params)
        layout.addWidget(self.TableParamsWidget)
        layout.addStretch()


    def save_data(self):
        return {**self.TableParamsWidget.save_data()}