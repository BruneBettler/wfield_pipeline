"""
denoising.py contains all the functions necessary for denoising wfield data.
"Denoising aims to isolate signal to boost signal-to-noise ratio".

Written by Brune Bettler
and Matthew Loukine
"""

import numpy as np
import scipy
from debug_visualize import frame_show
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tqdm import tqdm


def denoise_svd(im_array, rank):
    '''
    Single value decomposition

    :param im_array: np array containing stack of 2D wfield images
    '''
    SVD_stack = []
    for i, frame in tqdm(enumerate(im_array), desc="Denoising"):
        # Calculate U, S, and Vt
        U, S, VT = np.linalg.svd(frame, full_matrices=False)
        #print(U.shape)
        #print(S.shape)
        #print(VT.shape)
        '''Lines 27-32 from Brune'''
        # Remove sigma values below threshold (250)
        #S_clean = np.array([Si if Si > 1000 else 0 for Si in S])

        # Calculate A' = U * S_clean * V
        #denoised_frame = np.array(np.dot((U * S_clean), Vh))

        #Only using the components from rank <= to reconstruct image
        S = np.diag(S)
        denoised_frame = U[:,:rank] @ S[0:rank,:rank] @ VT[:rank,:]
        im_array[i] = denoised_frame
        SVD_stack.append([U[:,:rank],S[0:rank,:rank],VT[:rank,:]])

    return (np.array(im_array),np.array(SVD_stack))

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



