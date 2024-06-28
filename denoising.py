"""
denoising.py contains all the functions necessary for denoising wfield data.
"Denoising aims to isolate signal to boost signal-to-noise ratio".
"""

import numpy as np
import scipy
from debug_visualize import frame_show
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tqdm import tqdm


def denoise_svd(im_array):
    '''
    Single value decomposition

    :param im_array: np array containing stack of 2D wfield images
    :return: np array containing stack of denoised 2D wfield images
    '''
    denoised_stack = []
    for frame in tqdm(im_array, desc="Denoising"):
        # Calculate U, S, and Vt
        #TODO try with both scipi and numpy and see which is better!
        #U, S, Vh = np.linalg.svd(im_array, full_matrices=False)
        U,S,Vh = scipy.linalg.svd(frame, full_matrices=False)


        # Remove sigma values below threshold (250)
        S_clean = np.array([Si if Si > 1000 else 0 for Si in S])

        # Calculate A' = U * S_clean * V
        denoised_frame = np.array(np.dot((U * S_clean), Vh))

        denoised_stack.append(denoised_frame)

    return np.array(denoised_stack)

'''
Penalized matrix decomposition 
'''

'''
https://www.nature.com/articles/s41467-022-32886-w
'''


'''
https://www.ipol.im/pub/art/2019/226/article_lr.pdf
'''

'''
https://www.cise.ufl.edu/~arunava/papers/pami-denoising.pdf
'''

if __name__ == "__main__":
    im_stack = np.load("registered_stack.npy")
    single_im = im_stack[0][0]

    denoised_im = denoise_svd(single_im)

    difference = single_im - denoised_im
    frame_show([single_im, denoised_im, difference], ["noise", "denoised", "difference"])

    print("done")



