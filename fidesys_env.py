import os
import sys

def setup_fidesys():

    fidesys_path = r'C:\Program Files\Fidesys\CAE-Fidesys-8.1'
    prep_path = os.path.join(fidesys_path, 'preprocessor', 'bin')

    # ВАЖНО: менять окружение ЗДЕСЬ
    os.environ["PATH"] += os.pathsep + prep_path
    sys.path.append(prep_path)

    try:
        import cubit
        import fidesys
    except ModuleNotFoundError:
        print("Fidesys not found")
        raise

    cubit.init([""])

    fc = fidesys.FidesysComponent()
    fc.init_application(prep_path)
    fc.start_up_no_args()

    return cubit, fidesys, fc