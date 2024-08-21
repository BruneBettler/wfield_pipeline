# Modified Code

### Mesonet/mesonet/mask_functions: TestGenerator
- line 67:
    - original: `img = io.imread(os.path.join(test_path, img_list[i]))`
    - new: `img = io.imread(img_list[i])`
    - reason: `img_list[i]` contains paths of the form *Data_inputs/image_0.png* and `test_path` contains the path *Data_inputs* as such, the `os.path.join` command creates an incorrect path: *Data_inputs/Data_inputs/image_0.png*. 

- Line 74: 
    - same change and reason as above for line 67. 

### ...anaconda3\envs\DEEPLABCUT\lib\site-packages\mesonet\mask_functions.py
- line 67:
    - same change and reason as above. No need to edit the mesonet files as these have already been imported to the DEEPLABCUT file. 
    - make sure to edit any mesonet files through the DEEPLABCUT env ! 
- line 74: 

### ...PycharmProjects\MesoNet\mesonet\dlc/config.yaml
- line 7:
    - original: `project_path: C:\Users\user\Desktop\mesonet\atlas-DongshengXiao-2020-08-03`
    - new: `project_path: C:\Users\bbettl\PycharmProjects\MesoNet\mesonet\models\atlas-DongshengXiao-2020-08-03`
    - reason: incorrect path to model directory
- line 11: 
    - new:  C:\Users\bbettl\PycharmProjects\MesoNet\mesonet\models\atlas-DongshengXiao-2020-08-03\videos\DLC_atlas.avi:
    - reason: incorrect path 


### ...\AppData\Local\anaconda3\envs\DEEPLABCUT\lib\site-packages\mesonet\mask_functions.py
- line 954: added a line to make sure that the directory was made if it did not previously exist...

### ...l\PycharmProjects\MesoNet\mesonet\model.py
- line 101: changed input and outputs to inputs and outputs (model = Model(inputs=inputs, outputs=conv10))

### C:\Users\bbettl\AppData\Local\anaconda3\envs\DEEPLABCUT\lib\site-packages\mesonet\model.py
- same change as above line

### C:\Users\bbettl\AppData\Local\anaconda3\envs\DEEPLABCUT\lib\site-packages\napari_deeplabcut\_writer.py
- line 51, added a line that specifies the root variable as it was previously not present in the meta dictionary as a key (KeyError).
- `root = meta["project"] + r'\labeled-data'`

### ...anaconda3\envs\DEEPLABCUT\Lib\site-packages\mesonet\utils.py
- line 139: added a more explanatory print statement
- changed line to: `print(f"Git repo base path: {git_repo_base}")`

