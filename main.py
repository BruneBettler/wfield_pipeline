'''
Written by Brune
Last Edit: Monday June 03 2024
'''

from data_preprocessor import rawDataPreprocessor
from utils import get_recording_paths

'''
FILL OUT PARAMS DICTIONARY WITH DESIRED PREPROCESSING CHOICES
'''
params = {
    'path_to_session_data':       '/Volumes/MATT_1/wfield/14-May-2024',
    'pipe_num':           0,              # select desired pipeline number
    'get_raw_brightness': False,
    'get_DeltaF':         False,
    'get_zScore':         False
          }


def main():
    # given a path to a single session, retrieve all recordings
    session_recording_paths = get_recording_paths(params['path_to_session_data'])
    # create an obj for each recording within the session
    recordings = {}
    for n, session_path in enumerate(session_recording_paths):
        recordings[f'recording_{n}'] = rawDataPreprocessor(session_path)
        if n == 0:
            break

    print('done loading things into object')

    recordings[f'recording_0'].demo_pipeline()

if __name__ == '__main__':
    main()

