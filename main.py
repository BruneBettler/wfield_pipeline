'''
Written by Brune
Last Edit: June 28 2024

'''

from data_preprocessor import rawDataPreprocessor
from utils import get_recording_paths
import datetime

'''
FILL OUT PARAMS DICTIONARY WITH DESIRED PREPROCESSING CHOICES
'''
params = {
    'path_to_session_data':       r"D:\wfield\14-May-2024",
    'pipe_num':           0,              # select desired pipeline number
    'get_raw_brightness': False,
    'get_DeltaF':         False,
    'get_zScore':         False
          }


def main():
    print(f"{datetime.datetime.now().time()}: Starting main.py")
    # given a path to a single session, retrieve all recordings
    session_recording_paths = get_recording_paths(params['path_to_session_data'])
    # create an obj for each recording within the session
    recordings = {}
    for n, session_path in enumerate(session_recording_paths):
        recordings[f'recording_{n}'] = rawDataPreprocessor(session_path)
        if n == 0:
            break

    print(f'{datetime.datetime.now().time()}: Done loading session data into object')

    recordings[f'recording_0'].demo_pipeline()

if __name__ == '__main__':
    main()

