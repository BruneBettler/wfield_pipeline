'''
Written by Brune
Last Edit: Monday June 03 2024
'''

from data_preprocessor import rawDataPreprocessor

'''
FILL OUT PARAMS DICTIONARY WITH DESIRED PREPROCESSING CHOICES
'''
params = {
    'path_to_data':       '/Volumes/MATT_1/wfield/wfield',
    'pipe_num':           0,              # select desired pipeline number
    'get_raw_brightness': False,
    'get_DeltaF':         False,
    'get_zScore':         False
          }


def main():
    folder_path = params['path_to_data']
    preprocessor = rawDataPreprocessor(folder_path)
    preprocessor.demo_pipeline()

    print('done loading things into object')


if __name__ == '__main__':
    main()