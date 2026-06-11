from PySide6.QtWidgets import QMainWindow, QWidget, QMessageBox, QVBoxLayout, QTabWidget, QLabel, QCheckBox, QPushButton
from PySide6 import QtGui, QtCore, QtWidgets
from collections import defaultdict

class EmptyWindow(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        layout=QVBoxLayout(self)
        self.checkbox = QCheckBox("Рассматривать все комбинации")
        layout.addWidget(self.checkbox)
        layout.addStretch()
    def get_setting(self):
        return {"checkbox": self.checkbox.isChecked()}

class WindowforTable(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        self.saved_data=None
        layout = QtWidgets.QVBoxLayout()
        self.setWindowTitle("Таблица")
        self.setFixedSize(600, 300)
        self.Table_steps = WidgetforTable(self)
        self.Table_steps.vbox.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.Table_steps)
        self.save = QtWidgets.QPushButton("Сохранить и закрыть")
        self.save.clicked.connect(self.on_save)
        layout.addWidget(self.save)
        self.setLayout(layout)


    def get_data(self):
        if self.saved_data is not None:
            return self.saved_data
        else:
            QMessageBox.critical(self, "Внимание", "Вы не ввели ни одного значения")

    def on_save(self):

        self.saved_data=self.Table_steps.save_data()
        print(self.saved_data)
        self.accept()

    def set_params(self,params):
        self.Table_steps.table.params_on_table(params)

class WidgetforTable(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.vbox = QtWidgets.QVBoxLayout()
        self.label_table = QtWidgets.QLabel("Установите диапазон параметров")
        self.table = Table()

        # Настройка таблицы
        self.table.horizontalHeader().setStretchLastSection(True)  # Растягивать последнюю колонку
        self.table.setAlternatingRowColors(True)  # Чередование цветов строк

        # Устанавливаем ширину колонок
        self.table.setColumnWidth(0, 130)  # Первая колонка шире

        button_layout = QtWidgets.QHBoxLayout()
        btn_add = QPushButton("добавить столбец")
        btn_del = QPushButton("удалить столбец")
        self.vbox.addWidget(self.table)
        btn_add.clicked.connect(self.add_column)
        btn_del.clicked.connect(self.del_column)
        button_layout.addWidget(btn_add)
        button_layout.addWidget(btn_del)
        self.vbox.addLayout(button_layout)
        self.setLayout(self.vbox)

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

    def params_on_table(self, params):
        print(params)
        self.setRowCount(len(params))
        print(self.data)
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
        return {"ranges": self.data}


class StepSettingsWidget(QWidget):
    def __init__(self, params):
        super().__init__()
        self.setup_ui()
        self.params=params

    def setup_ui(self):
        layout=QVBoxLayout(self)

        table_button = QPushButton("Установить значения параметров")
        table_button.clicked.connect(self.show_table)
        label = QtWidgets.QLabel("Оптимизация по параметрам")
        layout.addWidget(label)
        layout.addWidget(table_button)

        layout.addStretch()
        self.empty_window=EmptyWindow()
        self.TableParamsWidget=WindowforTable()
        self.setMinimumHeight(352)

    def show_table(self):
        self.TableParamsWidget.set_params(self.params)
        self.TableParamsWidget.exec()



    def save_data(self):
        return {**self.TableParamsWidget.get_data()}