import time
import dill
import os
        
def save():
    now = time.strftime('%Y-%m-%d_%H:%M')
    try:
        os.mkdir('jupyterData')
    except FileExistsError:
        pass
    filename = 'jupyterData/%s.jupyterData' % now
    dill.dump_session(filename)
    print('data saved')

class NoJupyterData(Exception):
    pass

def load(newest=True, filename=None):
    files = [file for file in os.listdir('jupyterData') if file.endswith('.jupyterData')]
    if len(files) == 0:
        raise NoJupyterData
        
    file_dir = {os.path.getmtime(f'jupyterData/{file}'):file for file in files}
    if newest:
        newest_file = file_dir[max(file_dir)]
        dill.load_session(f'jupyterData/{newest_file}')
        print(f'{newest_file} is loaded')
    else:
        if filename is None:
            print(files)
            filename = input('choose one datafile to load:')
        try:
            dill.load_session(f'jupyterData{filename}')
            print(f'{filename} is loaded')
        except NameError:
            print('Check filename!')
            return -1      