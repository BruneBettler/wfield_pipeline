'''
Written by Brune
Last Edit: Monday June 03 2024
'''

import cv2
import numpy as np
from tqdm import tqdm


def motion_correct(dat, out=None, chunksize=512, nreference=60, mode='ecc', apply_shifts=True):
    '''
    Motion correction by translation.
    This estimate x and y shifts using phase correlation.

    The reference image is the average of the chunk in the center.

    Inputs:
        dat (array)           : (NFRAMES, NCHANNEL, H, W) is overwritten if apply_shifts is True
        chunksize (int)       : size of the chunks (needs to be small enough to fit in memory - default 512)
        nreference            : number of frames to take as reference (default 60)
        apply_shifts          : overwrite the data with the motion corrected (default True)
    Returns:
        yshifts               : shitfs in y (NFRAMES, NCHANNELS)
        xshifts               : shifts in x
    '''
    nframes, nchan, h, w = dat.shape
    if out is None:
        out = dat
    chunks = chunk_indices(nframes, chunksize)
    xshifts = []
    yshifts = []
    rshifts = []
    # reference is from the start of the file (nreference frames to nreference*2)
    # (chunksize frames and for each channel independently)
    nreference = int(nreference)
    chunk = np.array(dat[nreference:nreference * 2])
    refs = chunk[0].astype('float32')
    # align to the ref of each channel and use the mean
    _, refs = _register_multichannel_stack(chunk, refs, mode=mode)
    refs = np.mean(refs, axis=0).astype('float32')
    for c in tqdm(chunks, desc='Motion correction'):
        # this is the reg bit
        localchunk = np.array(dat[c[0]:c[-1]])
        # always do 2d first
        # (xs,ys,rot),corrected = _register_multichannel_stack(localchunk,refs,
        #                                                     mode='2d')
        # if not mode == '2d' :
        # xs += xs0
        # ys += ys0
        (xs, ys, rot), corrected = _register_multichannel_stack(localchunk, refs, mode=mode)
        if apply_shifts:
            out[c[0]:c[-1]] = corrected[:]
        yshifts.append(ys)
        xshifts.append(xs)
        rshifts.append(rot)
    return (np.vstack(yshifts), np.vstack(xshifts)), np.vstack(rshifts)

def chunk_indices(nframes, chunksize = 512, min_chunk_size = 16):
    '''
    Gets chunk indices for iterating over an array in evenly sized chunks
    '''
    chunks = np.arange(0, nframes, chunksize, dtype = int)
    if (nframes - chunks[-1]) < min_chunk_size:
        chunks[-1] = nframes
    if not chunks[-1] == nframes:
        chunks = np.hstack([chunks,nframes])
    return [[chunks[i],chunks[i+1]] for i in range(len(chunks)-1)]

def _register_multichannel_stack(frames, templates, mode='2d', niter = 25, eps0 = 1e-3,  warp_mode = cv2.MOTION_EUCLIDEAN): # mode 2d
    nframes, nchannels, h, w = frames.shape

    if mode == 'ecc':
        hann = cv2.createHanningWindow((w,h),cv2.CV_32FC1)
        hann = (hann*255).astype('uint8')

    ys = np.zeros((nframes, nchannels), dtype=np.float32)
    xs = np.zeros((nframes, nchannels), dtype=np.float32)
    rot = np.zeros((nframes, nchannels), dtype=np.float32)
    stack = np.zeros_like(frames, dtype='uint16')

    for ichan in range(nchannels):
        chunk = frames[:,ichan].squeeze()
        if mode == '2d':
            res = runpar(registration_upsample, chunk,
                         template = templates[ichan])
            ys[:,ichan] = np.array([r[0][1] for r in res],dtype='float32')
            xs[:,ichan] = np.array([r[0][0] for r in res],dtype='float32')

        elif mode == 'ecc':
            res = runpar(registration_ecc, chunk,
                         template = templates[ichan],
                         hann = hann,
                         niter = niter,
                         eps0 = eps0,
                         warp_mode = warp_mode)
            xy,rots = _xy_rot_from_affine([r[0] for r in res])
            ys[:,ichan] = xy[:,1]
            xs[:,ichan] = xy[:,0]
            rot[:,ichan] = rots
        stack[:,ichan,:,:] = np.stack([r[1] for r in res])
    return (xs,ys,rot), stack