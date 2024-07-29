'''
Written by Brune
Last Edit: June 28 2024
'''

from data_loading_functions import *
from registration import motion_correct
from denoising import denoise_svd
import datetime

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

        self.all_frames = load_dat_frames(filename=self.path_to_frame_data)

        self.blue_frames = self.all_frames[:, 0, ...] # blue frame occurs first
        self.violet_frames = self.all_frames[:, 1, ...]

        # TODO: add code to take care of the analog coordination things current i'm just doing the frame data

        self.analog_data = load_dat_analog(self.path_to_analog)
        self.frame_times = load_mat_frameTimes(self.path_to_frameTimes)

        self.blue_ftimes = self.frame_times['frameTimes'][::2] # blue frame occurs first
        self.violet_ftimes = self.frame_times['frameTimes'][1::2]

        # the following parameteres are for visualization / ... # TODO: write here more info on what's going on
        self.verbose = True  # change to False if you do not want printed updates as the code runs


    def demo_pipeline(self):
        # 1. Motion correction / registration
        print(f'{datetime.datetime.now().time()}: Starting Motion Correction')
        _, _, self.motion_corrected_frames_all = motion_correct(dat=self.all_frames, out=None, mode='ecc', apply_shifts=True)
        #motion_corrected_frames_all = np.load("registered_stack.npy")
        print(f'{datetime.datetime.now().time()}: Done Motion Correction')

        # The step is not needed as the data is low resolution,
        # noise should be reduced out, and compression is not needed, as data is not large.
        # In the future, important to remember this step and consider it!

        # 2. Denoising and Compression
        # print(f'{datetime.datetime.now().time()}: Starting  Blue Frames')
        # self.denoised_blue_frames, self.blue_SVD_stack = denoise_svd(motion_corrected_frames_all[:,0,...],rank=200) # only denoise blue channel for now
        # print(f'{datetime.datetime.now().time()}: Done Denoising Blue Frames')

        # print(f'{datetime.datetime.now().time()}: Starting  Violet Frames')
        # self.denoised_violet_frames, self.violet_SVD_stack = denoise_svd(motion_corrected_frames_all[:,1,...],rank=200) # only denoise blue channel for now
        # print(f'{datetime.datetime.now().time()}: Done Denoising Violet Frames')
        
        # 3. Hemocorrection 
        # This step is not the hemocorrection done by Churchland lab, but is
        # a simpler verision. 
        print(f'{datetime.datetime.now().time()}: Starting HemoCorrection')
        
                
        # 4. Segmentation
        #print(f'{datetime.datetime.now().time()}: Starting Segmentation')
        return denoised_frames_all

        return 0


if __name__ == "__main__":
    motion_corrected_frames_all = np.load("registered_stack.npy")
    # 2. Denoising
    channel_num = motion_corrected_frames_all.shape[1]
    all_chan_denoised = []
    for channel_i in range(channel_num):
        denoised_frames_all = denoise_svd(motion_corrected_frames_all[:,channel_i,...])
        all_chan_denoised.append(denoised_frames_all)

    print("done")









