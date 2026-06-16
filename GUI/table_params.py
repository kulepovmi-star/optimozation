from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QTabWidget, QLabel
from PySide6 import QtGui, QtCore, QtWidgets

class TableParamsWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.vbox = QtWidgets.QVBoxLayout()
        self.label_table = QtWidgets.QLabel("Установите диапазон параметров")
        self.vbox.setContentsMargins(0, 0, 0, 0)
        self.table = TableParams()
        self.table.setMinimumHeight(300)
        # Настройка таблицы
        self.table.horizontalHeader().setStretchLastSection(True)  # Растягивать последнюю колонку
        self.table.setAlternatingRowColors(True)  # Чередование цветов строк

        # Устанавливаем ширину колонок
        self.table.setColumnWidth(0, 130)  # Первая колонка шире
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)

        #  КНОПКИ УПРАВЛЕНИЯ ТАБЛИЦЕЙ
        button_layout = QtWidgets.QHBoxLayout()
        self.add_row_btn = QtWidgets.QPushButton("Очистить")
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
        data = self.table.get_data()
        return data

    def clean(self):
        self.table.del_item()


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
                    data[self.item(row, 0).text()].append(float(self.item(row, column).text()))
        return data

    def get_params(self):
        return self.params

    def set_params_on_table(self, data):
        for row, value in enumerate(self.params):
            item = QtWidgets.QTableWidgetItem(value)
            self.setItem(row, 0, item)  # только колонка 0
            item_min = QtWidgets.QTableWidgetItem(str(min(data[value])))
            self.setItem(row, 1, item_min)  # только колонка 0
            tem_max = QtWidgets.QTableWidgetItem(str(max(data[value])))
            self.setItem(row, 2, tem_max)  # только колонка 0

