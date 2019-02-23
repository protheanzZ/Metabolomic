import time
import dill

def save():
    now = time.strftime('%H %M-%d%h')
    dill.dump_session('%s.jupyterData' % now)
    

    