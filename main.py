import sys  # Cистемная библиотека
import os  # Cистемная библиотека
import GUI
from PySide6 import QtWidgets
from jou_reader import JouReader





if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'первый тест', 'балка')
    reader = JouReader(file_path)
    app=QtWidgets.QApplication(sys.argv)
    script, params = reader.read()
    widget=GUI.Dialog()
    widget.set_script(script)
    widget.set_params(params)
    widget.show()
    sys.exit(app.exec())


    #start(fidesys, aprepro, base_dir)

