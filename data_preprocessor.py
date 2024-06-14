'''
Written by Brune
Last Edit: Monday June 03 2024
'''

from data_loading_functions import *
from data_preprocessing_functions import *

class rawDataPreprocessor:
    def __init__(self, path_to_folder):
        '''
        Initializes a rawDataPreprocessor object given the path to a folder containing the following items for a single imaging session:
        Analog_1.dat, Frames_..._uint16_....dat, frameTimes_....mat, handles.mat (these 4 files minimum)
        '''
        if path_to_folder[-1] != '/':
            path_to_folder += '/'

        self.path_to_analog = get_file_path(path_to_folder, 'A')
        self.path_to_frame_data = get_file_path(path_to_folder, 'F')
        self.path_to_frameTimes = get_file_path(path_to_folder, 'T')

        all_frames = load_dat_frames(filename=self.path_to_frame_data)

        self.blue_frames = all_frames[:, 0, ...] # blue frame occurs first
        self.violet_frames = all_frames[:, 1, ...]

        # TODO: add code to take care of the analog coordination things current i'm just doing the frame data

        self.analog_data = load_dat_analog(self.path_to_analog)
        self.frame_times = load_mat_frameTimes(self.path_to_frameTimes)

    def demo_pipeline(self):
        # TODO: ADD STEPS HERE
        return 0








