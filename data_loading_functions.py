'''
Written by Brune
Last Edit: Monday June 03 2024
'''

import os
import numpy as np
from pylab import *
import struct
import scipy.io

def get_file_path(path_to_folder, file_type):
    '''
    Helper function for __init__
    file_type either "A" for analog, "F" for frames.dat, "T" for frameTimes.mat
    '''
    for _, _, files in os.walk(path_to_folder):
        for file_name in files:
            if file_type == 'A' and file_name[:6] == 'Analog' and file_name[-4:] == '.dat':
                file_path = path_to_folder + file_name
                return file_path
            if file_type == 'F' and file_name[:6] == 'Frames' and file_name[-4:] == '.dat':
                file_path = path_to_folder + file_name
                return file_path
            if file_type == 'T' and file_name[:10] == 'frameTimes' and file_name[-4:] == '.mat':
                file_path = path_to_folder + file_name
                return file_path
    return 0

def load_dat_frames(filename, nframes=None, offset=0, shape=None, dtype='uint16'):
    '''
    Loads frames from a binary file.

    Inputs:
        filename (str)       : fileformat convention, file ends in _NCHANNELS_H_W_DTYPE.dat
        nframes (int)        : number of frames to read (default is None: the entire file)
        offset (int)         : offset frame number (default 0)
        shape (list|tuple)   : dimensions (NCHANNELS, HEIGHT, WIDTH) default is None
        dtype (str)          : datatype (default uint16)
    Returns:
        An array with size (NFRAMES,NCHANNELS, HEIGHT, WIDTH).

    Example:
        dat = load_dat(filename)
    '''
    if not os.path.isfile(filename):
        raise OSError('File {0} not found.'.format(filename))
    if shape is None or dtype is None:  # try to get it from the filename
        dtype, shape, _ = _parse_binary_fname(filename, shape=shape, dtype=dtype)
    if type(dtype) is str:
        dt = np.dtype(dtype)
    else:
        dt = dtype

    if nframes is None:
        # Get the number of samples from the file size
        nframes = int(os.path.getsize(filename) / (np.prod(shape) * dt.itemsize))
    framesize = int(np.prod(shape))

    offset = int(offset)
    with open(filename, 'rb') as fd:
        fd.seek(offset * framesize * int(dt.itemsize))
        buf = np.fromfile(fd, dtype=dt, count=framesize * nframes)
    buf = buf.reshape((-1, *shape), order='C')

    return buf

def _parse_binary_fname(fname, lastidx=None, dtype='uint16', shape=None, sep='_'):
    '''
    Gets the data type and the shape from the filename
    This is a helper function to use in load_dat.

    out = _parse_binary_fname(fname)

    With out default to:
        out = dict(dtype=dtype, shape = shape, fnum = None)
    '''
    fn = os.path.splitext(os.path.basename(fname))[0]
    fnsplit = fn.split(sep)
    fnum = None
    if lastidx is None:
        # find the datatype first (that is the first dtype string from last)
        lastidx = -1
        idx = np.where([not f.isnumeric() for f in fnsplit])[0]
        for i in idx[::-1]:
            try:
                dtype = np.dtype(fnsplit[i])
                lastidx = i
            except TypeError:
                pass
    if dtype is None:
        dtype = np.dtype(fnsplit[lastidx])
    # further split in those before and after lastidx
    before = [f for f in fnsplit[:lastidx] if f.isdigit()]
    after = [f for f in fnsplit[lastidx:] if f.isdigit()]
    if shape is None:
        # then the shape are the last 3
        shape = [int(t) for t in before[-3:]]
    if len(after) > 0:
        fnum = [int(t) for t in after]
    return dtype, shape, fnum

def load_dat_analog(file_path):
    """
    Convert a .dat file to a numpy ndarray\n
    First we read the file header.
    The first data [double] is representing the amount of data in the header
    The second double is the time of acquisition onset on first run
    The Third double is the number of recorded analog channels + timestamps
    The Fourth or last double is the number of values to read (set to inf since absolute recording duration is unknown at this point)

    After the Data is written as uint16

    """
    DLB = 8
    UILB = 2

    with open(file_path, mode='rb') as file: #open the file as a binary file
        data = file.read()
    data_in_header = int(struct.unpack('d', data[:DLB])[0])                     #read the amount of the data in header
    header = []
    for i in range(DLB, DLB*data_in_header+1, DLB):                             # read the header
        info = struct.unpack('d', data[i:i+DLB])[0]
        header.append(info)
    number_of_chanels = int(header[1])
    # data_converted = []
    # for i in range((data_in_header+1) * DLB, len(data), UILB*number_of_chanels):   #Read the Data chunks
    #     buff = []
    #     for j in range(number_of_chanels):                                         #Read by chanel
    #         buff.append(struct.unpack('H', data[i+j*UILB:i+j*UILB+UILB])[0])
    #     data_converted.append(buff)
    # data_converted = numpy.array(data_converted)                                # Convert to ndarray
    data_shape =(int((len(data)-(data_in_header+1) * DLB)/UILB/number_of_chanels), number_of_chanels)
    data = data[(data_in_header+1) * DLB:]
    deserialized = np.frombuffer(data, dtype=np.uint16)
    reshaped = np.reshape(deserialized, newshape=data_shape)
    # print(f"Identical: {numpy.all(data_converted == reshaped)}")
    return reshaped

def load_mat_frameTimes(file_path): #
    '''
    function returns a dictionary containing the file data in frameTimes.mat
        'removedFrames': single value
        'frameTimes':
        'preStim':
        'postStim':
        'imgSize': 1,4 array
    '''

    data = scipy.io.loadmat(file_path)
    return data



if __name__ == "__main__":
    print("done")