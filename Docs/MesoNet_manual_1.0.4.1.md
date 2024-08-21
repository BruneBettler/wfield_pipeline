# MesoNet
# 1. Introduction
MesoNet is a comprehensive Python toolbox for automated registration and segmentation of mesoscale mouse brain images.
You can use it to:
- Automatically identify cortical landmarks
- Register a brain atlas to your mesoscale calcium activity data (or vice versa)
- Segment brain data based on a brain atlas
- Use one of our machine learning  models (or train your own) to identify and classify brain regions without using any landmarks!

We developed atlas-to-brain and brain-to-atlas approaches to make the software flexible, easy to use and robust.

We offer an easy to use GUI, as well as a powerful command line interface (CLI) allowing you to integrate the toolbox with your own neural imaging workflow.

We also extend our pipeline to make use of functional sensory maps and spontaneous cortical activity motifs. We developed novel animal-specific motif-based functional maps that represent cortical consensus patterns of regional activation that can be used for brain registration and segmentation.

We provided six end-to-end automated pipelines to allow users to quickly output results from input images. We provided Code Ocean capsules to demonstrate the operation of all these automated MesoNet pipelines at https://doi.org/10.24433/CO.1919930.v1, and https://doi.org/10.24433/CO.4985659.v1.


MesoNet is built primarily on the U-Net machine learning model
[(Ronneberger, Fischer, and Brox, 2015)](http://dx.doi.org/10.1007/978-3-319-24574-4_28),
as adapted in [zhixuhao](https://github.com/zhixuhao)'s [unet repository](https://github.com/zhixuhao/unet), as well as
[DeepLabCut](https://github.com/AlexEMG/DeepLabCut), [keras](https://github.com/keras-team/keras), and
[opencv](https://github.com/opencv/opencv).

## Installation
1. For DeepLabCut functionality (necessary for identifying brain landmarks!),
[install and set up a DeepLabCut environment](https://github.com/AlexEMG/DeepLabCut/blob/master/docs/installation.md).
We recommend using their supplied Anaconda environments.
2. Activate the environment (as described above, usually `activate DEEPLABCUT` on Windows or 
`source activate DLC-GPU` on Linux/Mac). 
3. Clone this git repository: `git clone https://github.com/bf777/MesoNet.git` (unless you already have a zipped version of the repository).
* NOTE: If you are on Windows, please clone the repository to a location on `C://` as the git repository search function does not currently support other drives.
4. Enter the git repository folder using `cd mesonet`, then run `python setup.py install` to install additional
dependencies for MesoNet. Alternatively, you can install the code by going to `MesoNet_wheel` at the top level of the repository, then running `pip install mesonet-1.0.4.1-py3-none-any.whl`.

Example data for input is available in the `tests/test_input` sub-folder of the repository, as well as in the sub-folders of the `Example_data` directory at the top level of the repository. In the `tests` folder, you can run `python test.py` to run the software from start to finish.

## Dependencies
MesoNet has the following dependencies, which should be automatically installed during the installation process described above:
```
'imutils', 'scikit-image', 'scipy', 'numpy==1.16.4', 'keras==2.3.1', 'opencv-python', 'Pillow',
```
```
'deeplabcut', 'pandas', 'matplotlib', 'python-polylabel', 'imgaug', 'voxelmorph'
```

## Supported platforms
MesoNet has been tested on Windows 8.1, Windows 10, and Linux (Ubuntu 16.04 and Arch 5.7); it should also work on older versions of Windows and on MacOS, but these platforms have not been tested. It works with or without a GPU, but a GPU is _strongly_ recommended for faster training and processing. MesoNet can be used with or without a GUI, and can be run on headless platforms such as Google Colab.

## Typical install and run times
Installation should take at most 25-30 minutes, assuming that DeepLabCut has not already been installed. Running cortex boundary and landmark estimation on a dataset should take at most 1-2 minutes on a CPU (likely faster on a GPU). Training a new cortex boundary or landmark estimation model can take anywhere from 1-3+ hours for satisfactory results on a GPU (longer on a CPU).

## Contributors
[Dongsheng Xiao](https://github.com/DongshengXiao) designed the processing pipeline, collected data, trained the DeepLabCut, U-Net and VoxelMorph models provided with MesoNet, and developed the brain atlas alignment approach. [Brandon Forys](https://github.com/bf777) wrote the code of GUI and CLI. They are in the [Murphy Lab](https://murphylab.med.ubc.ca/) in UBC’s Department of Psychiatry.

# 2. Quick Start Guide
Ready to start using MesoNet? This is the place to begin!

**Note:** Before starting this tutorial, make sure you've installed MesoNet and its dependencies using the instructions found in [Installation](#installation).

**Note 2:** All external links worked as of the writing of this codebook, but some may change or no longer work. Please visit the [MesoNet website](https://github.com/bf777/MesoNet) for the latest information.

MesoNet can be used in one of two ways:

* through a **graphical user interface (GUI)**, which doesn't require you to type any code. This method is useful if you have a set of brain images and you want to predict and export the brain regions from each image, and aren't interested in customizing various parameters such as the machine learning model parameters or the brain atlases used to make predictions.

* through a **command line interface (CLI)**, which requires you to type a minimal amount of code using IPython. This method is useful if you're using a Jupyter notebook, Colab environment, or another environment that doesn't easily support GUIs, or if you want to customize various parameters in the code.

Additionally, MesoNet can be used through five approaches:
1. **Atlas to brain**: Given a pre-trained DeepLabCut model that was trained to associate anatomical landmarks with corresponding points on atlases of brain regions, register an atlas of brain regions to the fixed brain imaging data using affine transformations. This approach is useful if your data has common anatomical landmarks and is the most robust to variations in image quality and orientation within your data.
2. **Brain to atlas**: Given a pre-trained DeepLabCut model that was trained to associate anatomical landmarks with corresponding points on atlases of brain regions, the brain imaging data to a fixed atlas of brain regions using affine transformations. This approach is useful if you would like to normalize your brain images to a common template based on anatomical landmarks.
3. **Atlas to brain + sensory maps**: Given a pre-trained DeepLabCut model that was trained to associate anatomical landmarks with corresponding points on atlases of brain regions as well as a set of folders containing functional brain activity for that animal that is consistent across animals, register an atlas of brain regions to the fixed brain imaging data using affine transformations. This approach is useful if you have consistent peaks of functional activity across animals that you would like to use in the alignment processes.
4. **Motif-based functional maps (MBFMs) + U-Net**: Given a pre-trained U-Net model that was trained to associate brain imaging data with atlases of brain regions, predict the locations of brain regions in the data without the use of landmarks. The brain imaging data should be motif-based functional maps (MBFMs) calculated using the associated MATLAB code (https://doi.org/10.24433/CO.4985659.v1). This approach is useful if one wishes to mark functional regions based on more complex features of the data (e.g. a motif-based functional map) than landmarks.
5. **Motif-based functional maps (MBFMs) + Brain-to-atlas + VoxelMorph**: Given a pre-trained VoxelMorph model that was trained to compute a non-linear transformation between a template functional brain atlas and brain image data, predict the locations of brain regions in the data. In particular, this approach can register each input brain image to a user-defined template functional atlas. The brain imaging data should be motif-based functional maps (MBFMs) calculated using the associated MATLAB code (using seqNMF). This approach is useful if your images are consistently oriented and you want to compare the predicted locations of brain regions across different images.

Let's go through how you can use each method to predict and export brain regions:

## Preparation
If you are using U-Net (.hdf5) or VoxelMorph (.h5) models, make sure that these models are placed in a new folder called “models” within the mesonet subdirectory of the MesoNet git repository.

Place all of your brain images in a single directory. For best results, please make sure that all of your brain images are 8-bit, and in .png format.

Alternatively, you can analyze an image stack in .tif format. Simply place the .tif image stack in an otherwise empty folder, and MesoNet will analyze all images in the stack.

The GUI method is outlined right below; you can also [skip to the command line method](#command-line-interface-method) if you prefer to work in a coding environment.

***

## Graphical User Interface (GUI) method

### Quick usage reference:

1. 
```
activate DLC-GPU
```
2.
```
ipython
```
3.
```
import mesonet
mesonet.gui_start()
```
4. Input folder: select input images
5. OPTIONAL: Inspect images using arrow keys/buttons
6. Output folder: select folder to which you'll output images
7. Select U-Net model to use in list on right
8. OPTIONAL: Select options and landmarks to use
9. Click "Predict brain regions using landmarks"
10. Check outputs (segmented brain image in GUI, other outputs in selected output folder)

### Complete guide:

1. If you haven't already done so, open your favourite command line application (e.g. Terminal on Mac or Linux, or Command Prompt on Windows) and activate the DeepLabCut environment for your computer by typing `activate` followed by the name of the DeepLabCut environment that you installed (based on [the instructions given here](https://github.com/AlexEMG/DeepLabCut/blob/master/conda-environments/README.md)). For example, you might type:
```
activate DLC-GPU
```
2. Start IPython by typing `ipython` at the command line.
3. To start the GUI for applying an existing model to your dataset, type:
```
import mesonet
mesonet.gui_start()
```

**NOTE**: Running the GUI using the above command will automatically search for your MesoNet git repository so that it can access necessary non-Python files (e.g. masks, U-Net models, etc.). On some computers (especially Linux) this process can take a long time. If you are experiencing very long (>1 min) load times when starting the GUI, you can manually define the git repository by using the command `mesonet.gui_start(git_repo='path/to/repo')`, where `'path/to/repo'` is a string containing the full path to the top level of your MesoNet git repository (i.e. the folder containing `mesonet`, `setup.py`, etc. You can change the git repository used at any time from within the GUI as well with the git repository browse button.

4. Next to "Input folder" at the top of the screen, click "Browse..." and select a folder containing your brain images (.png, .npy, and .mat will work) or a single .tif, .npy, or .mat image stack, which should be in the format outlined above in the Preparation section. Don't be alarmed if nothing appears in the directory - if you have images matching these criteria in the directory, they will appear in the GUI!

5. Next to "Save folder" at the top of the screen, click "Browse..." and select (or create) an empty folder to which you want to save your analyses.

6. For "DLC config folder", if you are using your own DeepLabCut model to identify landmarks on the mouse cortex, locate the config.yaml file for that model here. Otherwise, the system will use a default model included with MesoNet (at the path shown in the box at "DLC config folder" when you start up MesoNet). IMPORTANT: for this step to work, you must navigate in the MesoNet repository to `mesonet/dlc/config.yaml`, and change the file paths after `project_path` and `video_sets` to the path to the displayed folder _on your own computer_. **NOTE: you will likely have to change the path here to find a config.yaml file for the model you wish to use!**

You can now use the arrow buttons at the bottom of the screen, or the left and right arrow keys on your keyboard, to browse through all of the images that MesoNet will analyze.

7. Now you can configure the settings on the right side of the screen:

* Select the U-Net model that you want to use to find the edges of the cortex (important if you've trained more than one model!) If you're looking for a model and can't find it here, check the `models` folder in the MesoNet git repository and ensure that your desired .hdf5 model is there. IMPORTANT: for this step to work, you must make sure that the `mesonet/models` folder in the MesoNet repository contains the `DongshengXiao_bundary.hdf5` file within the `Unet_model` folder within the repository.

* Select "Save predicted regions as .mat files" if you want to export each brain region as a region of interest (ROI) as a MATLAB (.mat) file. Select this if you have a workflow that involves detecting activity in specific brain regions through MATLAB or Octave (e.g. overlaying an ROI on functional brain imaging to identify activity in a specific brain region).

* Select "Use U-Net for alignment" if you have a U-Net model selected from the list above and you wish to constrain the aligned atlas to within the borders of the visible cortex. **This enables the U-Net approach**.

* Select "Use DeepLabCut for alignment" if you have a DeepLabCut model and you wish to align the atlas and brain based on landmarks defined in this model. **This enables the Landmark Estimation (DeepLabCut) approach**.

* Select "Use VoxelMorph for alignment" if you have a VoxelMorph model and wish to align brain regions within the borders of the cortex based on a provided template. **This enables the VoxelMorph approach**.

* Select "Draw olfactory bulbs" if the olfactory bulbs are visible in all images.

* Select "Align atlas to brain" if you wish to register a standardized atlas to the brain images. Uncheck this option if, instead, you wish to normalize and align the brain images to a stationary standardized atlas. The latter approach may allow the brain regions to be identified in a more consistent manner, but requires transformation back to the native space of the brain for follow-up analyses if you do not want to work in normalized space. 

* Select "Align using sensory map" *only* if you have images of functional activity in the brain, and a set of coordinates for peaks of functional activity overlaid on a brain atlas that corresponds to these images. This will allow you to use your functional activity data to potentially improve the quality of the brain region predictions. If you have such images, create a folder with one subfolder for each brain image you plan to analyze. For example, if you're analyzing images 0, 1, 2, 3, ... create subfolders 0, 1, 2, 3, ... Within each subfolder, place three images of functional activity for that specific brain. The peaks of functional activity in each image should ideally represent regions that are consistently activated given a specific activity (e.g. whisker stimulation). Make sure to locate the folder containing your sensory map subfolders under "Sensory map folder".

* Select "Plot DLC landmarks on final image" to plot the landmarks as predicted by DeepLabCut as *large* circles, and the the landmarks on the aligned atlas as *small circles*. For best results, points of corresponding colour should be as close to each other as possible.

* Select "Align based on first brain image only" to only calculate a transformation (brain-to-atlas or atlas-to-brain) based on the first brain image only (as opposed to individually for each brain image). This can save time if all of your brain images are from the same animal and are perfectly aligned. Additionally, if you want to initially align each brain image based on a template (i.e. the first image in your set), but then conduct all other follow-up alignments using each individual image, choose this option.

* Select "Use old label consistency method (less consistent)" to try and assign labels to brain regions by brain hemisphere. This method may not assign labels consistently between brain images because it is partly dependent on the positioning of the contours. The method used when this box is not checked requires a brain atlas (in .csv format) that has each brain region filled with a unique numeric label; it provides very high consistency in the labels of the brain regions between brain regions.

* The remaining nine check-boxes allow you to select the landmarks to be used in the alignment. You can align with as few as two landmarks or as many as nine landmarks; the default (from the included, default DeepLabCut model) is nine landmarks, which are as follows (with stereotaxic coordinates relative to bregma, in mm):
| Position | Definition | Coordinates (mm) |
| -------- | :--------- | :-------------- |
|1. **Left** | Anterolateral tip of the left parietal bone | (-3.13, 2.19) |
|2. **Top left** | Left frontal pole | (-1.83, 3.41) |
|3. **Bottom left** | Posterior tip of the left retrosplenial region | (-0.85, -4.02) |
|4. **Top centre** | Cross point between the median line and the line which connects the left and right frontal pole | (0, 3.41) |
|5. **Bregma** | Bregma | (0, 0) |
|6. **Lambda** | Anterior tip of the interparietal bone | (0, -3.49) |
|7. **Right** | Anterolateral tip of the right parietal bone | (3.13, 2.19) |
|8. **Top right** | Right frontal pole | (1.83, 3.41) |
|9. **Bottom right** | Posterior tip of the right retrosplenial region | (0.85, -4.02) |

For best alignment results, try and select as many landmarks as are present in your data, or at least two points in each hemisphere (left and right), as well as at least two of `Top centre`, `Bregma`, and `Lambda`. If you leave all of the landmarks selected, the ones used for alignment by default will be `Left`, `Top centre`, `Lambda`, and `Top right`. If you deselect any of these four landmarks, the first three landmarks you selected in the left hemisphere and the last three landmarks you selected in the right hemisphere will be used. For example, if you deselect landmarks 1 (`Left`) and 9 (`Bottom right`), the landmarks that will be used are 2, 3, and 4 in the left hemisphere and 8, 7, and 6 in the right hemisphere.

* Click on "Open VoxelMorph settings" to define options for the VoxelMorph stage of the workflow, which carries out a second, model-based transformation of local functional regions based on a pre-trained model. Take a look at the window that appears (it may appear behind the current image).
* Next to "Template file location", select the folder containing a template file to which the brain image will be registered. This will usually be a functional brain image containing functional motifs to which you want to align an input functional image. This image can be in .png, .mat, or .npy format.
* Only change "Flow file location" if you want to apply transformations based on an existing VoxelMorph deformation field (such as the .npy file that is output alongside each atlas in `output_mask`. If you define an existing flow file to use, make sure to check "Use existing transformation" in order to use that flow file.
* In the box to the right, select the VoxelMorph model that will be used to compute the transformation. Any models you add should be placed in the `models/voxelmorph` subfolder of the MesoNet git repository; they will then appear in this list.

* When you've defined your input images, a save folder, and U-Net model (and optionally a set of sensory maps), click "Predict brain regions using landmarks". Select this option to automatically predict brain regions in all of your brain images. 

* The numerical labels on the resulting brain image correspond to the order in which MesoNet identified the brain regions in your image. In general, brain regions in each image which have the same number are considered by MesoNet to be the same brain region. If you selected the "Save predicted regions as .mat files" checkbox, the numbers on those files also correspond to the brain regions you see here. 
* Leave "Predict brain regions directly using pretrained U-Net model" alone unless you have a machine learning model (selected in the white box above) that you've trained to segment your specific brain images. If you do have such a model, click this button to predict the shape and location of each brain region in your brain image based on your model.

* You can browse through the segmented brain images using the left and right arrow buttons at the bottom of the screen, or the left and right arrow keys on your keyboard.

* You can also use MesoNet as a quick interface for evaluating animal behaviour using DeepLabCut. First, place the .yaml config file for your DeepLabCut pose estimation model in `dlc -> behavior` in the `mesonet` subfolder. Next, select a set of images that you'd like to analyze for a behavior in the Behavior Input folder at the top of the screen; select a folder to which you'd like to save these images using the Behavior Save folder; then click the "Predict animal movements" button on the right. Please note that this feature is experimental.

8. Congrats, you're done! You can now go to the save folder that you selected and find all of the data files and products of this analysis, including the images output by each step, the landmark predictions, and your .mat files (if you chose to generate them). In the save folder, you'll also find a file called 'mesonet_test_config.yaml'; this is a file containing all of the settings you defined for this analysis in the GUI, and you can use this config file in the command line method described below to extend your analysis in the command line.

***

## Command Line Interface method

If you want to have more control over the parameters of MesoNet's analysis, or simply feel more comfortable working with a bit of IPython, then we also offer a straightforward command line interface.

### Quick usage reference:
* Save brain images as outlined in [Preparation](#preparation)
```
activate [your DeepLabCut environment, e.g. DLC-GPU]
```
```
ipython
```
```
import mesonet
```
```
input_file = 'path/to/input/folder'
```
```
output_file = 'path/to/output/folder'
```
```
config_file = mesonet.config_project(input_file, output_file, 'test')
```
```
mesonet.predict_regions(config_file)
```
```
mesonet.predict_dlc(config_file)
```

* If you want to rerun or continue an existing analysis (e.g. continue an analysis you started in the GUI):
```
import mesonet
```
```
config_file = 'path/to/config/file'
```
```
mesonet.predict_regions(config_file)
```
```
mesonet.predict_dlc(config_file)
```

### Complete guide:

1. If you haven't already done so, open your favourite command line application (e.g. Terminal on Mac or Linux, or Command Prompt on Windows) and activate the DeepLabCut environment for your computer by typing `activate` followed by the name of the DeepLabCut environment that you installed (based on [the instructions given here](https://github.com/AlexEMG/DeepLabCut/blob/master/conda-environments/README.md)). For example, you might type:
```
activate DLC-GPU
```
2. Type `ipython` to enter the IPython interpreter, then import the MesoNet package by typing `import mesonet`.
3. First, we need to define the folder containing your input brain images:
```
input_file = 'path/to/input/folder'
```
where path/to/input/folder is the folder containing your input images (make sure they're in the format discussed above in [Preparation](#preparation)). If you're on Windows, make sure to add an r before the first single quote (e.g. `r'C:\...'`).
4. Now, define the folder to which you'll save the output brain images:
```
output_file = 'path/to/input/folder'
```
5. MesoNet's command line interface works using configuration files that define - and allow you to customize - settings for each analysis (see section 4 for a full guide on all customizable parameters). To generate a config file for your analysis, run:
```
config_file = mesonet.config_project(input_file, output_file, 'test')
```
The 'test' command indicates that this config file will be used to apply an existing model to predict brain regions. We'll be adding the ability to train a new brain region segmentation model through this method soon! This step will generate a config file in the `output_file` directory (i.e. save directory) that you defined. You can open this file (`mesonet_test_config.yaml`) with any text editor; the details of its parameters can be found in the [Config File Guide](#config-file-guide) below.

NOTE: the parameters `use_dlc`, `use_unet`, and `use_voxelmorph` will activate the **Landmark Estimation (DeepLabCut) approach**, the **U-Net approach**, and the **VoxelMorph approach**, respectively, if set to `True`.

If you didn't type 'config_file =' before the command above, define the path to this config file by running:
```
config_file = 'path/to/config/file'
```
where `path/to/config/file` is the full path to the config file in your save directory.

6. The analysis itself runs in two steps. Firstly, to generate the masks of the brain's outline to be used in the second stage of the analysis, run:
```
mesonet.predict_regions(config_file)
```
After running this step, you may wish to look at your save folder. It will now have a couple of new folders, one of which is called `output_mask`. This folder contains the masks of the brain's outline.

7. Lastly, run:
```
mesonet.predict_dlc(config_file)
```
to generate the segmented brain regions! These will be saved in the `output_overlay` folder in your save folder.

# 3. Model Training

Ready to train your own MesoNet model to identify brain regions? Start here!

**Note:** Before starting this tutorial, make sure you've installed MesoNet and its dependencies using the instructions found in [Installation](#installation).

As with the testing interface, MesoNet can be used in one of two ways:

* through a **graphical user interface (GUI)**, which doesn't require you to type any code. This method is useful if you have a set of brain images and you want to paint in the cortex region using a graphical interface, and aren't interested in customizing various parameters such as the machine learning model parameters.

* through a **command line interface (CLI)**, which requires you to type a minimal amount of code using IPython. This method is useful if you're using a Jupyter notebook, Colab environment, or another environment that doesn't easily support GUIs, or if you want to customize various parameters in the code. This option is also useful if you already have a set of binary masks paired with the brain images (e.g. created in another image editing app).

Let's go through how you can use each method to train a model to automatically segment a cortical image by region.

## Basic procedure
### U-Net
The U-Net model will delimit the boundary of the cortex visible in the brain image. As such, in order to train this model, you will need to pair each input brain image with a drawing where the cortex is white and everything outside the cortex is black. Our GUI provides you with an interface to create and save these drawings alongside the original input brain images.

### DeepLabCut
The DeepLabCut model will locate nine cortical landmarks based on your labelling these landmarks in a set of brain images. Our GUI helps you prepare the DeepLabCut project, guides you into DeepLabCut's labelling GUI, and trains the DeepLabCut model based on these labels.

## Preparation
For best results, please make sure that all of your brain images are 8-bit, and in .png format.

**GUI method**: Place all of your brain images in a single directory.

**Command line interface method**: For each image that you wish to use in your training set, create a binary mask where the whole cortical region is filled in white and the rest of the picture is filled in black. Put the brain images in a folder called `image`, and put the masks in a folder called `label`. Put both of these folders together in a single folder, which you'll use as your input folder.

***

## Graphical User Interface (GUI) method

### Quick usage reference:

1. 
```
activate DLC-GPU
```
2.
```
ipython
```
3.
```
import mesonet
mesonet.gui_start('train')
```
4. Input folder: select input images
5. Save folder: select folder to which you'll save the input images paired with the U-Net masks
6. Log folder: select folder to which you'll record output data
7. DLC folder: select folder to which you'll output the DeepLabCut model
8. Paint a mask of the cortical boundaries, then click "Save current mask to file"
9. Choose model name (type an existing model name to update existing model with new data)
10. Click "Train U-Net model" to train model
11. Create a DeepLabCut configuration file with "Generate DLC config file"
12. Click "Label brain images with landmarks" to label DeepLabCut landmarks
13. Set up parameters to train DeepLabCut model, then click "Train DLC model"

### Complete guide:

1. If you haven't already done so, open your favourite command line application (e.g. Terminal on Mac or Linux, or Command Prompt on Windows) and activate the DeepLabCut environment for your computer by typing `activate` followed by the name of the DeepLabCut environment that you installed (based on [the instructions given here](https://github.com/AlexEMG/DeepLabCut/blob/master/conda-environments/README.md)). For example, you might type:
```
activate DLC-GPU
```

2. Start IPython by typing `ipython` at the command line.

3. To start the GUI for training a new model to your dataset, type:
```
import mesonet
mesonet.gui_start('train')
```

4. In the interface that appears, next to "Input folder" at the top of the screen, click "Browse..." and select a folder containing your brain images, which should be in the format outlined above in the Preparation section. Don't be alarmed if nothing appears in the directory - if you have images matching these criteria in the directory, they will appear in the GUI! For example data for the U-net model training, you can use the images in `Training_set/U-Net_model_data`. For example data for the DeepLabCut model training, you can use the images in `landmark_estimation_model_data`.

5. Next to "Save folder" at the top of the screen, click "Browse..." and select (or create) an empty folder to which you want to save your brain images and masks for training the model.

6. Next to "Log folder" at the top of the screen, click "Browse..." and select (or create) an empty folder to which you want to record the output data from the U-Net model.

7. Next to "DLC folder" at the top of the screen, click "Browse..." and select (or create) an empty folder to which you want to save the DeepLabCut project.

8. Turn your attention to the brain image. As discussed in [Preparation](#Preparation), you will need to paint in the cortical region on the brain image. You can adjust the brush size for the painting (in pixels) by changing the value next to "Brush Size" in the top right part of the window. Click and drag on the image where you want to paint the cortical region (white pixels). Once you're finished, click "Save current mask to file". This will save your painted region against a black background into a `label` subfolder of the directory you chose in "Save folder". A copy of the original image will also be saved into a corresponding `image` subfolder. Go through each image using the left and right arrow keys or the arrow buttons at the bottom of the screen, painting in the cortical region and clicking "Save current mask to file" each time. If you want to repaint an image, you can return to that image and paint it over again; saving it again will overwrite the original image.

9. Give your model a name by typing it in the box next to "Model name"; make sure the model name ends with `.hdf5`. NOTE: If the model name that you select already exists in your copy of the MesoNet git repository, it will train the existing model with the data that you have provided (i.e. online learning) instead of training a new model. You may wish to do this if, for example, you have a good model and you want to train it on additional data to make it more robust. Click on "Train U-Net model" to train the U-Net model. 

Note that your input images and masks will be automatically augmented (rotated and skewed) to increase the robustness of the model. Note that at the end of training, you may need to exit the GUI. In that case, simply re-enter IPython and type:
```
import mesonet
mesonet.gui_start('train')
```
before filling in the input, save, log, and DLC folders. You do not need to re-create the masks, as they've already been saved.

10. Congratulations, you've created your U-Net model! Now it is time to label the landmarks on the brain for processing in DeepLabCut (DLC). Type a label for the model in the box next to "Task", and type the labeler's name next to "Name". Then, click "Generate DLC config file" to generate a new folder where the DLC computations will take place (this folder will be located at the path specified in the "DLC Folder" box). The folder will be labelled in the format `Task-Labeler-YYYY-MM-DD` (e.g. `MesoNet-Labeler-2020-04-29`).

11. Click "Label brain images with landmarks" to be taken to DeepLabCut's interface for labeling the landmarks. Note that you may have to type "yes" in the command line after you click this button. You can find instructions on how to use this interface from the DeepLabCut developers [here](https://github.com/AlexEMG/DeepLabCut/blob/master/docs/functionDetails.md#d-label-frames).

12. Once you've finished labeling the images, check the parameters under "Display iters", "Save iters", and "Max iters". These numbers define how often DeepLabCut will print out the results of the model, at what frequency it should save the model (e.g. every 1000 iterations), and (in newer versions of DLC) the maximum number of iterations for which the model should run. Once you've adjusted these settings as desired, click on "Train DLC model" to train the DeepLabCut model! Note that at the end, you may once again have to quit the GUI when you want to exit the model (by pressing Control-C).

## Command Line Interface method

If you want to have more control over the parameters of MesoNet's analysis, or simply feel more comfortable working with a bit of IPython, then we also offer a straightforward command line interface.

### Quick usage reference:
* Save brain images as outlined in [Preparation](#preparation)
```
activate [your DeepLabCut environment, e.g. DLC-GPU]
```
```
ipython
```
```
import mesonet
```
```
input_file = 'path/to/input/folder'
```
```
output_file = 'path/to/output/folder'
```
```
config_file = mesonet.config_project(input_file, output_file, 'train')
```
* OPTIONAL: inspect config file to change information for model training
```
mesonet.train_model(config_file)
```
* Use DeepLabCut to train a model with landmark locations on a dataset of brain images

### Complete guide:

1. If you haven't already done so, open your favourite command line application (e.g. Terminal on Mac or Linux, or Command Prompt on Windows) and activate the DeepLabCut environment for your computer by typing `activate` followed by the name of the DeepLabCut environment that you installed (based on [the instructions given here](https://github.com/AlexEMG/DeepLabCut/blob/master/conda-environments/README.md)). For example, you might type:
```
activate DLC-GPU
```
2. Type `ipython` to enter the IPython interpreter, then import the MesoNet package by typing `import mesonet`.
3. First, we need to define the folder containing your input brain images:
```
input_file = 'path/to/input/folder'
```
where path/to/input/folder is the folder containing your input images (make sure they're in the format discussed above in [Preparation](#preparation)). If you're on Windows, make sure to add an r before the first single quote (e.g. `r'C:\...'`).
4. Now, define the folder to which you'll save the output brain images:
```
output_file = 'path/to/input/folder'
```
5. MesoNet's command line interface works using configuration files that define - and allow you to customize - settings for each analysis. To generate a config file for your analysis, run:
```
config_file = mesonet.config_project(input_file, output_file, 'train')
```
The 'train' command indicates that this config file will be used to define the parameters for a new model. This step will generate a config file in the `output_file` directory (i.e. save directory) that you defined. You can open this file (`mesonet_train_config.yaml`) with any text editor; the details of its parameters can be found in the [Training config file](#training-config-file) section of the [Config File Guide](#config-file-guide) below.

If you didn't type 'config_file =' before the command above, define the path to this config file by running:
```
config_file = 'path/to/config/file'
```
where `path/to/config/file` is the full path to the config file in your save directory.

6. To train a model based on your pairing of brain images and corresponding masks according to your selected options in the config file, run:
```
mesonet.train_model(config_file)
```
After running this step, you may wish to look at your save folder. It will now have output information about the model training process. The model itself, by default, is saved to the `models` folder of the MesoNet git repository, where it will be automatically detected and selectable as a model for use in your analyses.

7. Use DeepLabCut to train a model based on a dataset of brain images, which you can label with your desired number of landmarks for later use in MesoNet. You can find more details on how to train a DeepLabCut model [here](https://github.com/DeepLabCut/DeepLabCut/blob/master/docs/UseOverviewGuide.md).

# 4. Config File Guide

If you're using the [command line](#command-line-interface-method) to run MesoNet, you will generate a configuration file containing various parameters that can be adjusted to customize your MesoNet analysis.

### Testing config file

When you are _testing_ a MesoNet model, MesoNet will output a config file called `mesonet_test_config.yaml` to your chosen output folder. Here's a guide to the parameters you can change in that file:

`atlas` (default: `False`): Set to `True` to just predict the cortical landmarks on your brain images, and not segment your brain images by region. Upon running `mesonet.predict_dlc(config_file)`, MesoNet will output your brain images labelled with these landmarks as well as a file with the coordinates of these landmarks. Set to `False` to carry out the full brain image segmentation workflow.

`config` (default: `dlc/config.yaml`): The location (within the MesoNet repository) of a DeepLabCut config file that references the landmark alignment model that we provide with MesoNet. Don't change this unless you have your own DeepLabCut model for identifying cortical landmarks that you would like to use instead.

`git_repo_base` (automatically determined): The automatically determined location of your MesoNet git repository, which is necessary to access the U-Net and DLC models that we provide. Do not change this path manually unless you're moving the git repository around after creating this config file (the repository will automatically be located each time you generate this config file).

`input_file` (no default): The full path to the folder containing your input brain images. This must be specified as an argument to `mesonet.config_project` when you create this config file, but you can change the path here at any time.

`mat_save` (default: `True`): Choose whether or not to export each predicted cortical region, each region's centrepoint, and the overall region of the brain to a .mat file (`True` = output .mat files, `False` = don't output .mat files).

`model` (default: `models/unet_bundary.hdf5`): The location (within the MesoNet repository) of a U-Net model to be used for finding the boundaries of the brain region (as the default model does), or (if you have a specially trained model for this purpose) segmenting the entire brain into regions without the need for atlas alignment. Only choose another model if you have another model that you would like to use for segmenting the brain.

`num_images` (automatically determined from `input_file`): The number of brain images in the folder. Only change this if you add or remove images from the folder after generating this config file but before starting your analysis (in particular, it's not a good idea to change this number after running `mesonet.predict_regions(config_file)` as it can cause the code to break).

`output_file` (no default): The full path to the folder to which you'd like to save all of the outputs of your MesoNet analyses. This must be specified as an argument to `mesonet.config_project` when you create this config file, but you can change the path here at any time.

`sensory_match` (default: `False`): If `True`, MesoNet will attempt to align your brain images using peaks of sensory activation on sensory maps that you provide in a folder named `sensory` inside your input images folder. If you do not have such images, keep this value as `False`.

`sensory_path` (no default): The full path to the folder containing the functional images optionally used for further aligning each input brain image. Ensure that this folder contains one folder for each input brain image, with three sensory images from that particular brain in each subfolder.

`threshold` (default: `0.0001`): Adjusts the sensitivity of the algorithm used to define individual brain regions from the brain atlas. 
**NOTE:** Changing this number may significantly change the quality of the brain region predictions; only change it if your brain images are not being segmented properly! In general, increasing this number causes each brain region contour to be smaller (less like the brain atlas); decreasing this number causes each brain region contour to be larger (more like the brain atlas).

`olfactory_check`: If True, draws olfactory bulb contours on the brain image.

`use_unet`: Choose whether or not to identify the borders of the cortex using a U-Net model.

`use_dlc`: Choose whether or not to try and register the atlas and brain image using a DeepLabCut model.

`atlas_to_brain_align`: If True, registers the atlas to each brain image. If False, registers each brain image to the atlas.

`plot_landmarks`: If True, plots DeepLabCut landmarks (large circles) and original alignment landmarks (small circles) on final brain image.

`align_once`: if True, carries out all alignments based on the alignment of the first atlas and brain. This can save time if you have many frames of the same brain with a fixed camera position.

`original_label`: if True, uses a brain region labeling approach that attempts to automatically sort brain regions in a consistent order (left to right by hemisphere, then top to bottom for vertically aligned regions). This approach may be more flexible if you're using a custom brain atlas (i.e. not one in which region is filled with a unique number).

`use_voxelmorph`: Choose whether or not to apply a local deformation registration for image registration, using a voxelmorph model.

`exist_transform`: if True, uses an existing voxelmorph transformation field for all data instead of predicting a new transformation.

`voxelmorph_model`: the name of a .h5 model located in the models folder of the git repository for MesoNet, generated using voxelmorph and containing weights for a voxelmorph local deformation model.

`template_path`: the path to a template atlas (.npy or .mat) to which the brain image will be aligned in voxelmorph.

`flow_path`: the path to a voxelmorph transformation field that will be used to transform all data instead of predicting a new transformation if exist_transform is True.

`coords_input_file`: The path to a file with DeepLabCut coordinates based on which a DeepLabCut transformation should be carried out.

`atlas_label_list`: A list of aligned atlases in which each brain region is filled with a unique numeric label. This allows for consistent identification of brain regions across images. If original_label is True, this is an empty list.

`model`: The location (within the MesoNet repository) of a U-Net model to be used for finding the boundaries of the brain region (as the default model does), or (if you have a specially trained model for this purpose) segmenting the entire brain into regions without the need for atlas alignment. Only choose another model if you have another model that you would like to use for segmenting the brain. 

`git_repo_base` (automatically determined): The automatically determined location of your MesoNet git repository, which is necessary to access the U-Net and DLC models that we provide and is the default save location for the U-Net models that you train. Do not change this path manually unless you're moving the git repository around after creating this config file (the repository will automatically be located each time you generate this config file).

`region_labels`: If True, MesoNet will attempt to label each brain region according to the Allen Institute's Mouse Brain Atlas. Otherwise, MesoNet will label each region with a number. Please note that this feature is  experimental! 

`landmark_arr`: The default number and order of landmarks to be used for the full alignment of a standard brain atlas to a brain image. Change what is contained in this list to change the landmarks used. When the default DeepLabCut model with nine landmarks, the landmarks are:

*	0: Anterolateral tip of the left parietal bone.
*	1: Left frontal pole.
*	2: Posterior tip of the left retrosplenial region.
*	3: Cross point between the median line and the line which connects the left and right frontal pole.
*	4: Bregma (centre point of cortex)
*	5: Anterior tip of the interparietal bone.
*	6: Anterolateral tip of the right parietal bone.
*	7: Right frontal pole.
*	8: Posterior tip of the right retrosplenial region.

`align_once`: If True, aligns all brains based on the alignment for the first brain. This can save time if all brain images in a sequence of images are of the same animal and have the same orientation and zoom.

`steps_per_epoch`: During U-Net training, the number of steps that the model will take per epoch. Defaults to  300 steps per epoch. 

`epochs`: During U-Net training, the number of epochs for which the model will run. Defaults to 60 epochs (set  lower for online learning, e.g. if augmenting existing model).

### Training config file

When you are _training_ a MesoNet model, MesoNet will output a config file called `mesonet_train_config.yaml` to your output folder. Here's a guide to the parameters you can change in that file:

`git_repo_base` (automatically determined): The automatically determined location of your MesoNet git repository, which is necessary to access the U-Net and DLC models that we provide and is the default save location for the U-Net models that you train. Do not change this path manually unless you're moving the git repository around after creating this config file (the repository will automatically be located each time you generate this config file).

`epochs` (default: `60`): The number of epochs for which the model will be trained. You may wish to reduce this number if you're updating an existing model.

`input_file` (no default): The full path to the folder containing your input brain images and corresponding binary masks delineating the borders of the brain region for each image. The brain images themselves must be in a subfolder of `input_file` called `images`; the masks must be in a subfolder called `label`. This must be specified as an argument to `mesonet.config_project` when you create this config file, but you can change the path here at any time.

`log_folder` (no default): The folder to which logging data should be output (and where this config file will be saved).

`model_name` (default: `my_unet.hdf5`): The name for the model that you are going to train (must include `.hdf5` extension!)

`steps_per_epoch` (default: `300`): The number of steps per training epoch. Influences the pace at which the model is trained.

`bodyparts` (default: ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']): The names and numbers of landmarks (bodyparts in DeepLabCut) to label in the DeepLabCut model.

`rotation_range`, `width_shift_range`, `height_shift_range`, `shear_range`, `zoom_range`, `horizontal_flip`, `fill_mode`: Keras image augmentation parameters for U-Net model training. See the [Keras documentation](https://keras.io/api/preprocessing/image/) for full documentation.

# 5. Troubleshooting

Below is a list of some issues that may arise while using MesoNet, and how to address them. If you experience other issues that are not addressed here, feel free to open an issue on the repository.

## Graphical User Interface (GUI)
1. Some of the buttons are greyed out! Why?
* The buttons for running the analysis are greyed out until you define an input folder containing the brain images, and a save folder to which you'll save your outputs.

2. When I selected my input folder, the screen remained blank!
* If this happens, make sure that you navigated to a folder which has your brain images directly inside it (i.e. not inside another folder in the folder you selected).

3. When I clicked a button to run my analysis, the GUI seemed to freeze!
* This is a side effect of the way in which Python handles user interfaces; the interface should unlock as soon as your analysis is done.

4. When I could use the GUI again, it didn't show me the segmented brain images!
* This is typically because MesoNet cannot find the output folder containing your segmented brain images. In this case, you can view the images manually by going to the save folder you defined and checking the `output_overlay` folder.

## Command Line Interface
1. When I run `mesonet.predict_dlc(config_file)`, it says that it can't find my config file!
* This may happen if MesoNet did not correctly locate the git repository for MesonNet (in which the default DeepLabCut configuration file is located). You can manually define the git repository location by going to the MesoNet configuration file (defined at `config_file`) and typing a path for `git_repo_base`.

2.  When I run `mesonet.predict_dlc(config_file)`, it says that it can't find a mask!
* This may happen if you did not run `mesonet.predict_regions(config_file)` first - as a result, the masks defining the boundary of each brain image may not have been generated. Try running that command first before running `mesonet.predict_dlc(config_file)`.

***

# Appendix A: Function reference

# Function reference

The following is the documentation for the functions that are accessible through MesoNet's command line interface. 

## GUI tools
```
mesonet.gui_start(gui_type='test')
```

Starts the MesoNet GUI.

**Parameters**:

`gui_type` (string): Default is 'test'. If 'test', opens the GUI for applying existing U-Net and DeepLabCut models to register and segment a set of brain images. If 'train', opens the GUI for training a new U-Net model (for automatically identifying the borders of the cortex in a brain image) and DeepLabCut model (for automatically identifying cortical landmarks for brain atlas - brain registration).

`git_repo` (string): Default is ''. If not '', the directory path supplied here will be used as the path to the MesoNet git repository. This can be useful to pre-define if your computer takes a while to find the repository automatically (this may occur on Unix systems).

`config_file` (string, **not functional**): Default is ''. If not '', the GUI will autofill based on information from the config file at the supplied file path. Note that this feature is still under construction.

## CLI tools

```
mesonet.config_project(input_dir, output_dir, mode, model_name='unet.hdf5', config='dlc/config.yaml', 
```
```
atlas=False, sensory_match=False, sensory_path='sensory', mat_save=True, use_unet=True, 
```
```
atlas_to_brain_align=True, olfactory_check=True, plot_landmarks=True, align_once=True, original_label=False,
```
```
threshold=0.0001, model='models/unet_bundary.hdf5',  region_labels=False, steps_per_epoch=300,
```
```
epochs=60)
```

Generates a config file (mesonet_train_config.yaml or mesonet_test_config.yaml, depending on whether you are applying an existing model or training a new one).

**Parameters**:

`input_dir`: The directory containing the input brain images.

`output_dir`: The directory containing the output files .

`mode`: If 'train', generates a config file for training; if 'test', generates a config file for applying  the model.

`model_name`: (optional) Set a new name for the unet model to be trained. Default is 'unet.hdf5' .

`config`: Select the config file for the DeepLabCut model to be used for landmark estimation. 

`atlas`:  Set to True to just predict the cortical landmarks on your brain images, and not segment your brain images by region. Upon running mesonet.predict_dlc(config_file), MesoNet will output your brain images  labelled with these landmarks as well as a file with the coordinates of these landmarks. Set to False to carry out the full brain image segmentation workflow. 

`sensory_match`: If True, MesoNet will attempt to align your brain images using peaks of sensory activation on sensory maps that you provide in a folder named sensory inside your input images folder.  If you do not have such  images, keep this value as False. 

`sensory_path`: If sensory_match is True, this should be set to the path to a folder containing sensory maps for each brain image. For each brain, put your sensory maps in a folder with the same name as the brain image (0, 1,  2, ...).

`mat_save`: Choose whether or not to export each predicted cortical region, each region's centre point, and the overall region of the brain to a .mat file (True = output .mat files, False = don't output .mat files).

`threshold`:  Adjusts the sensitivity of the algorithm used to define individual brain regions from the brain atlas. NOTE: Changing this number may significantly change the quality of the brain region predictions; only change it if your brain images are not being segmented properly! In general, increasing this number causes each brain region contour to be smaller (less like the brain atlas); decreasing this number causes each brain region contour to be larger (more like the brain atlas). 

`olfactory_check`: If True, draws olfactory bulb contours on the brain image. 

`use_unet`: Choose whether or not to identify the borders of the cortex using a U-Net model.

`use_dlc`: Choose whether or not to try and register the atlas and brain image using a DeepLabCut model.

`atlas_to_brain_align`: If True, registers the atlas to each brain image. If False, registers each brain image to the atlas. 

`plot_landmarks`: If True, plots DeepLabCut landmarks (large circles) and original alignment landmarks (small  circles) on final brain image. 

`align_once`: if True, carries out all alignments based on the alignment of the first atlas and brain. This can save time if you have many frames of the same brain with a fixed camera position.

`original_label`: if True, uses a brain region labelling approach that attempts to automatically sort brain regions in a consistent order (left to right by hemisphere, then top to bottom for vertically aligned regions). This approach may be more flexible if you're using a custom brain atlas (i.e. not one in which region is filled with a unique number).

`use_voxelmorph`: Choose whether or not to apply a local deformation registration for image registration, using a voxelmorph model.

`exist_transform`: if True, uses an existing voxelmorph transformation field for all data instead of predicting a new transformation.

`voxelmorph_model`: the name of a .h5 model located in the models folder of the git repository for MesoNet, generated using voxelmorph and containing weights for a voxelmorph local deformation model.

`template_path`: the path to a template atlas (.npy or .mat) to which the brain image will be aligned in voxelmorph.

`flow_path`: the path to a voxelmorph transformation field that will be used to transform all data instead of predicting a new transformation if exist_transform is True.

`coords_input_file`: The path to a file with DeepLabCut coordinates based on which a DeepLabCut transformation should be carried out.

`atlas_label_list`: A list of aligned atlases in which each brain region is filled with a unique numeric label. This allows for consistent identification of brain regions across images. If original_label is True, this is an empty list.

`model`: The location (within the MesoNet repository) of a U-Net model to be used for finding the boundaries  of the brain region (as the default model does), or (if you have a specially trained model for this purpose) segmenting the entire brain into regions without the need for atlas alignment. Only choose another model if you have  another model that you would like to use for segmenting the brain.

`region_labels`: If True, MesoNet will attempt to label each brain region according to the Allen Institute's Mouse Brain Atlas. Otherwise, MesoNet will label each region with a number. Please note that this feature is  experimental! 

`steps_per_epoch`: During U-Net training, the number of steps that the model will take per epoch. Defaults to  300 steps per epoch. 

`epochs`: During U-Net training, the number of epochs for which the model will run. Defaults to 60 epochs (set lower for online learning, e.g. if augmenting existing model).

This function returns `config_file`: The path to the config_file. If you run this function as config_file = config_project(...) then  you can directly get the config file path to be used later.

```
mesonet.predict_regions(config_file)
```
Segments brain regions using a U-Net model, based on parameters supplied in a .yaml configuration file.

**Parameters**:

Please see the [Config File Guide](#config-file-guide) for a full list of parameters that can be supplied in the .yaml configuration file.

```
mesonet.predict_dlc(config_file)
```
Predicts the locations of cortical landmarks on the brain, then uses these cortical landmarks (and, optionally, the U-Net model) to register a standard brain atlas to the brain image (or brain image to atlas).

**Parameters**:

Please see the [Config File Guide](#config-file-guide) for a full list of parameters that can be supplied in the .yaml configuration file.
