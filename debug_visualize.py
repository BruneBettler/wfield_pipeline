import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation


def frame_show(images_array, title_array):
    '''
    Function visualizes a single image given as a 2D array or a list of multiple images
    '''
    im_num = len(images_array)

    if im_num != 1:
        fig, axs = plt.subplots(1, im_num, figsize=((im_num * 6), 6))
        for im_i in range(im_num):
            axs[im_i].imshow(images_array[im_i])
            axs[im_i].set_title(title_array[im_i])
        plt.show()
    else:
        plt.imshow(images_array[0])
        plt.colorbar()
        plt.show()
        plt.title(title_array[0])
    return 0

def calculate_reg_error(ref_frame, registered_frames):
    '''
    Function calculates the registration error based on the ___metric
    '''

    '''
    Could do RMSE (lower value = better)
    could do MAE (lower value = better) 
    could to normalized cross-correlation (NCC) (higher = better) 
    automated landmark detection 
    mutual information (MI) (higher = better) statistical dependence 
    '''

    return 0

def show_film(orig_frames_array, reg_frames_array, title):
    num_frames, height, width = orig_frames_array.shape

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Initialize the plots
    def init():
        ax1.imshow(orig_frames_array[0]) # , cmap='gray'
        ax2.imshow(reg_frames_array[0]) # , cmap='gray'
        ax1.set_title(f'{title}: Pre-Registration')
        ax2.set_title(f'{title}: Post-Registration')
        return []

    # Update the plots for each frame
    def update(frame):
        ax1.imshow(orig_frames_array[frame]) #, cmap='gray'
        ax2.imshow(reg_frames_array[frame]) # , cmap='gray'
        return []

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=True)

    # Save the animation as a movie file
    ani.save('/Volumes/MATT_1/wfield/blue_frames.mp4', writer='ffmpeg', fps=30)

    # Show the animation in a window
    #ani.show()

    return 0


if __name__ == "__main__":
    orig_stack = np.load("original_stack.npy")
    orig_blue = np.array(orig_stack[:,0,...])
    orig_violet = np.array(orig_stack[:,1,...])
    reg_stack = np.load("registered_stack.npy")
    reg_blue = np.array(reg_stack[:,0,...])
    reg_violet = np.array(reg_stack[:,1,...])

    print("done here")

    #show_film(orig_blue, reg_blue, "Blue Frames")

    b = np.sqrt(np.mean((orig_blue-reg_blue)**2))
    v = np.sqrt(np.mean((orig_violet-reg_violet)**2))

    print(f'B = {b} \nV = {v}')

