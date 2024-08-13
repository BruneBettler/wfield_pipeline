'''
Written by Brune
Last Edit: June 28 2024
'''

from data_loading_functions import *
from registration import motion_correct
from denoising import denoise_svd
from hemocorrection import hemocorrection
import datetime
import pickle
import scipy.io

class rawDataPreprocessor:
    def __init__(self, path_to_folder, blue_and_violet_exists=True, stim_computer=True):
        '''
        Initializes a rawDataPreprocessor object given the path to a folder containing the following items for a single imaging session:
        Analog_1.dat, Frames_..._uint16_....dat, frameTimes_....mat, handles.mat (these 4 files minimum)
        '''
        if path_to_folder[-1] != '/':
            path_to_folder += '/'

        self.path_to_analog = get_file_path(path_to_folder, 'A')
        self.path_to_frame_data = get_file_path(path_to_folder, 'F')
        self.path_to_frameTimes = get_file_path(path_to_folder, 'T')

        # The following is added for olfactometer data
        if stim_computer:
            self.path_to_stimcomputer = get_file_path(path_to_folder, 'M')

        self.all_frames = load_dat_frames(filename=self.path_to_frame_data)
        
        if blue_and_violet_exists:
            self.blue_frames = self.all_frames[:, 0, ...] # blue frame occurs first
            self.violet_frames = self.all_frames[:, 1, ...]

        # # TODO: add code to take care of the analog coordination things current i'm just doing the frame data

        self.analog_data = load_dat_analog(self.path_to_analog)
        self.frame_times = load_mat_frameTimes(self.path_to_frameTimes)

        if blue_and_violet_exists:
            self.blue_ftimes = self.frame_times['frameTimes'][::2] # blue frame occurs first
            self.violet_ftimes = self.frame_times['frameTimes'][1::2]

        # the following parameteres are for visualization / ... # TODO: write here more info on what's going on
        self.verbose = True  # change to False if you do not want printed updates as the code runs

        # # The following is added for olfactometer data
        # if stim_computer:
        #     '''
        #     The olfactometer data found on the olfactometer computer is put into a .mat file. There is also an hd5 file.
        #     Here we use a .mat file. The .mat file contains either dictionaries in bit form, or arrays. The .mat file 
        #     has the following dicionaries : configs, experiment_start_timestamp, parameters, software_envrionment,
        #     stimulus_frame_info, stimulus_frame_info_text, use_data. It has the following arrays sync (double). sync_sc is
        #     16 bit, and sync_scaling a struct.

        #     configs - has the configuration for the experiment, such as stimulus duration, post and pre delays. etc.
        #     experiment_start_timestamp - the global computer time the experiment started
        #     parameters - has the computers paramters for the experiment
        #     software_environement - has information about the software for Acquisition
        #     stimulus_frame_info - contains start and end time of stimulus onset
        #     stimulus_frame_info_text - contains the same information as above
        #     sync - contains TTL information. [0] - time, [1] TTL odor pulses, [2] ____ ,[3] start time, [4] end time,
        #     sync_sc (same information)
        #     sync_scaling - scaling sync to sync_sc
        #     user_data - Can contain extra information from the aquistion software. Currently says which order is being shown in the order. 

        #     The code below does not get all the data out of the mat file, only the most important parts. Note for dictionaries, we must use pickle.loads()
        #     after getting the mat file loaded. Rather the arrays only use scipy.io.loadmat(path)['name']
        #     '''
        #     # TODO : put the following into its own function.
        #     self.olfac_experiment_start_timestamp = pickle.loads(scipy.io.loadmat(self.path_to_olfacdata)['experiment_start_timestamp'].tostring())
        #     self.olfac_stim_frame_info = pickle.loads(scipy.io.loadmat(self.path_to_olfacdata)['stimulus_frame_info'].tostring())
        #     self.olfac_user_data = pickle.loads(scipy.io.loadmat(self.path_to_olfacdata)['user_data'].tostring())

        #     self.olfac_sync = scipy.io.loadmat(self.path_to_olfacdata)['sync']


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
        print(f'{datetime.datetime.now().time()}: Starting  Blue Frames Denoising and Compression')
        self.denoised_blue_frames, _ = denoise_svd(motion_corrected_frames_all[:,0,...],rank=200)
        print(f'{datetime.datetime.now().time()}: Done Denoising Blue Frames')
        print(f'{datetime.datetime.now().time()}: Starting  Violet Frames Denoising and Compression')
        self.denoised_violet_frames, _ = denoise_svd(motion_corrected_frames_all[:,1,...],rank=200)
        print(f'{datetime.datetime.now().time()}: Done Denoising Violet Frames')
        
        # 3. Hemocorrection 
        # This step is not the hemocorrection done by Churchland lab, but is
        # a simpler verision. 
        print(f'{datetime.datetime.now().time()}: Starting HemoCorrection')
        self.hemo_corr_frames = hemocorrection(blue_frames=self.denoised_blue_frames, violet_frames=self.denoised_violet_frames)
        print(f'{datetime.datetime.now().time()}: Done HemoCorrection')
        
                
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









