'''
Written by Brune
Last Edit: Thursday June 06 2024
'''

from multiprocessing import Pool, cpu_count
from functools import partial
import os

def get_recording_paths(session_path):
    '''
    :param session_path: path to a single session folder containing different recordings
    :return: an array containing the paths of each recording folder within the inputted session
    '''
    # check if session_path ends with '/'
    if session_path[-1] != '/':
        session_path += '/'

    recording_paths = []
    for dirpath, dirnames, filenames in os.walk(session_path):
        for dir in dirnames:
            recording_paths.append(session_path+dir)
        return recording_paths

    return 1

def parinit():
    import os
    os.environ['MKL_NUM_THREADS'] = "1"
    os.environ['OMP_NUM_THREADS'] = "1"


def runpar(f,X,nprocesses = None,**kwargs):
    '''
    res = runpar(function,          # function to execute
                 data,              # data to be passed to the function
                 nprocesses = None, # defaults to the number of cores on the machine
                 **kwargs)          # additional arguments passed to the function (dictionary)

    '''

    if nprocesses is None:
        nprocesses = cpu_count()
    with Pool(initializer = parinit, processes=nprocesses) as pool:
        res = pool.map(partial(f,**kwargs),X)
    pool.join()
    return res


if __name__ == "__main__":
    paths = get_recording_paths('/Volumes/MATT_1/wfield/14-May-2024/')