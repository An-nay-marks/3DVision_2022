# 3D Face Reconstruction from Video
Project for 3D Vision 2022 at ETH Zurich, reconstruction pipeline for 3D face models from video.

## Setup
### Base environment
To create and activate the conda environment, use
```
conda env create -f environment.yml
conda activate 3DVision
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```
To install PyTorch3D, follow [INSTALL.md](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md).
If you encounter an import error in the keras-vggface package, you will need to change the import statement from
```python
from keras.engine.topology import get_source_inputs
```
to 
```python
from keras.utils.layer_utils import get_source_inputs
```
Unfortunately, this is a problem with the module and cannot be fixed by us.

### Submodules
Please clone all submodules.
If you plan on running the pipeline on CPU, you have to change one line of code in the DECA submodule due to different torch versions:
Goto dependencies/DECA/decalib/deca.py
Change line 89 to: `checkpoint = torch.load(model_path, map_location=torch.device(self.device))`

### Model checkpoints
Checkpoints for SCRFD, YOLOv5Face and ArcFace are provided in this repository.
Follow the instructions in [README](https://github.com/YadiraF/DECA/blob/master/README.md) under 
Usage > Preppare Data and copy the FLAME, DECA and Albedo files 
into [data/model_files](data/model_files).

### NoW  
[optional]
In order to run the NoW evaluation, download the data set from its [website](https://now.is.tue.mpg.de/). Then,
follow the instructions in the [README](https://github.com/soubhiksanyal/now_evaluation/blob/main/README.md)
to install additional dependencies.

## Usage
### Pipeline
The pipeline is divided into three main components: detection, classification and reconstruction. Each of these build
on the previous and will therefore also run all the stages leading up to it. They can be started 
from [main.py](main.py) or from their specific files as seen below.
```
python3 main.py -f {detection, classification, reconstruction} -s SOURCE [-r NAME] [--online]
```
This is equivalent to running one of the three stages directly.
```
python3 detect.py [-d {scrfd, yolo5}] [--patch-size int int]
```
```
python3 classify.py [-c {agglomerative, dbscan, mean-shift, vgg}] [-lr PATH]
```
```
python3 reconstruct.py [--merge {single, mean, mean_shape, 'cnn'}] [-lc PATH]
```
As each stage runs all previous stages, arguments are cumulative.

#### --source / -s
Specifies the path to the source video. Required unless --load-raw or --load-classified are used.

#### --run-name / -r
Can be used to specify a name for the current run, which will be used for the export directory. 
If unspecified, a name will be generated from the current time and date. 

#### --online
Binary option to specify whether to run the online or offline pipeline. The online pipeline will perform real-time 
classification and show the video while processing detected faces.

#### --detector / -d
Specify the model to use for the detection stage. Defaults to SCRFD.

#### --patch-size
Specify the export resolution of detected faces. Patches will not be resized if unspecified.

#### --classifier / -c
Specify the model / clustering algorithm used in the classification stage. Defaults to agglomerative clustering.

#### --load-raw / -lr
Specify a path to previously exported detection results. This allows skipping the first stage.

#### --merge
Specify the multi-face merging strategy used during reconstruction to obtain results from multiple 
images of the same person. Defaults to single reconstruction per image (no merging).

#### --load-classified / -lc
Specify a path to previously exported classification results. This allows skipping the first two stages.

### Model training
In order to be able to train the predictive quality model (CNN), the [now_dist.npy](data/now_dist.npy) file can be used. 
It includes DECA's NoW scores for all images in the data set + augmentation. If the data changes,
all requirements for NoW (see above) need to be present to regenerate it. Training can be started with
```
python3 optimize_deca.py [--epochs NUM_EPOCHS] [--checkpoint CHECKPOINT_PATH]
```
This can be done without the [now_dist.npy](data/now_dist.npy) file, but will be slow. To regenerate the file, run
```
python3 now_validation.py [--merge MERGE_STRATEGY]
```
Data augmentation on the NoW data set can be performed using
```
python3 augmentation.py
```
After training, retrieve the relevant model checkpoint from [checkpoints](checkpoints) and copy it into
[data/model_files](data/model_files) as 'conv_predictor.pt'.

### GUI
The graphical user interface can be launched with
```
python3 gui.py
```
The GUI has only been confirmed to work on Windows.
