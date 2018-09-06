# BodyNet: Volumetric Inference of 3D Human Body Shapes

[Gül Varol](http://www.di.ens.fr/~varol/), [Duygu Ceylan](http://www.duygu-ceylan.com/), [Bryan Russell](http://bryanrussell.org/), [Jimei Yang](https://research.adobe.com/person/jimei-yang/), [Ersin Yumer](http://www.meyumer.com/), [Ivan Laptev](http://www.di.ens.fr/~laptev/) and [Cordelia Schmid](http://lear.inrialpes.fr/~schmid/),
*BodyNet: Volumetric Inference of 3D Human Body Shapes*, ECCV 2018.

[[Project page]](http://www.di.ens.fr/willow/research/bodynet/) [[arXiv]](https://arxiv.org/abs/1804.04875)

<p align="center">
<img src="http://www.di.ens.fr/willow/research/bodynet/images/bodynet.png" title="BodyNet: Volumetric Inference of 3D Human Body Shapes" height="200", style="max-width:25%;vertical-align:top" /> &emsp;&emsp;&emsp;&emsp;
<img src="http://www.di.ens.fr/willow/research/bodynet/images/bodynet.gif" title="BodyNet: Volumetric Inference of 3D Human Body Shapes" height="200", style="max-width:25%;"/>
</p>

## Contents
* [1. Preparation](https://github.com/gulvarol/bodynet#1-preparation)
* [2. Training](https://github.com/gulvarol/bodynet#2-training)
* [3. Testing](https://github.com/gulvarol/bodynet#3-testing)
* [4. Fitting SMPL model](https://github.com/gulvarol/bodynet#4-fitting-smpl-model)
* [Citation](https://github.com/gulvarol/bodynet#citation)
* [Acknowledgements](https://github.com/gulvarol/bodynet#acknowledgements)

## 1. Preparation
 
### 1.1. Requirements
* Datasets
  * Download [SURREAL](https://github.com/gulvarol/surreal#1-download-surreal-dataset) and/or [Unite the People (UP)](http://files.is.tuebingen.mpg.de/classner/up/) dataset(s)
* Training
  * Install [Torch](https://github.com/torch/distro) with [cuDNN](https://developer.nvidia.com/cudnn) support.
  * Install [matio](https://github.com/soumith/matio-ffi.torch) by `luarocks install matio`
  * Install [OpenCV-Torch](https://github.com/VisionLabs/torch-opencv) by `luarocks install cv`
  * Tested on Linux with cuda v8 and cudNN v5.1.
* Pre-processing and fitting python scripts
  * Python 2 environment with the following installed:
    * [OpenDr](https://github.com/mattloper/opendr)
    * [Chumpy](https://github.com/mattloper/chumpy)
    * [OpenCV](https://pypi.org/project/opencv-python/)
  * SMPL related
    * Download [SMPL for python](http://smpl.is.tue.mpg.de/) and set `SMPL_PATH`
      * Fix the naming: `mv basicmodel_m_lbs_10_207_0_v1.0.0 basicModel_m_lbs_10_207_0_v1.0.0`
      * Do the following changes in the code `smpl_webuser/verts.py`:
      ```diff
      - v_template, J, weights, kintree_table, bs_style, f,
      + v_template, J_regressor, weights, kintree_table, bs_style, f,
      - if sp.issparse(J):
      -     regressor = J
      -     J_tmpx = MatVecMult(regressor, v_shaped[:,0])
      -     J_tmpy = MatVecMult(regressor, v_shaped[:,1])
      -     J_tmpz = MatVecMult(regressor, v_shaped[:,2])
      + if sp.issparse(J_regressor):
      +     J_tmpx = MatVecMult(J_regressor, v_shaped[:,0])
      +     J_tmpy = MatVecMult(J_regressor, v_shaped[:,1])
      +     J_tmpz = MatVecMult(J_regressor, v_shaped[:,2])
      +     assert(ischumpy(J_regressor))
      -     assert(ischumpy(J))
      + result.J_regressor = J_regressor
      ```
    * Download [neutral SMPL model](https://github.com/classner/up/blob/master/models/3D/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl) and place under `models` folder of SMPL
    * Download [SMPLify](http://smplify.is.tue.mpg.de/) and set `SMPLIFY_PATH`
  * Voxelization related
    * Download [binvox executable](http://www.patrickmin.com/binvox/) and set `BINVOX_PATH`
    * Download [binvox python package](https://github.com/dimatura/binvox-rw-py) and set `BINVOX_PYTHON_PATH`

### 1.2. Pre-processing for training
#### SURREAL voxelization
Loop over the dataset and run `preprocess_surreal_voxelize.py` for each `_info.mat` file by setting it with the `--input` option (for foreground and/or part voxels with the `--parts` option). The surface voxels are filled with `imfill` with the `preprocess_surreal_fillvoxels.m` script, but you could do it in python (e.g. `ndimage.binary_fill_holes(binvoxModel.data)`). Sample preprocessed data is included in `preprocessing/sample_data/surreal`.

#### Preparing UP data
Loop over the dataset by running `preprocess_up_voxelize.py` to voxelize and to re-organize the dataset. Fill the voxels with `preprocess_up_fillvoxels.m`. Preprocess the segmentation maps with `preprocess_up_segm.m`. Sample preprocessed data is included in `preprocessing/sample_data/up`.

### 1.3. Setup paths for training
Place the data under `~/datasets/SURREAL` and `~/datasets/UP` or change the `opt.dataRoot` in opts.lua. The outputs will be written to `~/cnn_saves/<datasetname>/<experiment>`, you can change the `opt.logRoot` to change the `cnn_saves` location.

### 1.4. Download pre-trained models
We provide several pre-trained models used in the paper [bodynet.tar.gz (980MB)](https://lsh.paris.inria.fr/bodynet/bodynet.tar.gz). The content is explained in the [training section](https://github.com/gulvarol/bodynet#2-training). Extract the `.t7` files and place them under `models/t7` directory.
``` bash
# Trained on SURREAL
model_segm_cmu.t7
model_joints3D_cmu.t7
model_voxels_cmu.t7
model_voxels_FVSV_cmu.t7
model_partvoxels_FVSV_cmu.t7
model_bodynet_cmu.t7
# Trained on UP
model_segm_UP.t7
model_joints3D_UP.t7
model_voxels_FVSV_UP.t7
model_voxels_FVSV_UP_manualsegm.t7
model_bodynet_UP.t7
# Trained on MPII
model_joints2D.t7
```

## 2. Training
There are sample scripts under `training/exp/backup` directory. These were created automatically using the `training/exp/run.sh` script. For example the following `run.sh` script:
``` bash
source create_exp.sh -h

input="rgb"
supervision="segm15joints2Djoints3Dvoxels" 
inputtype="gt"
extra_args="_FVSV"
running_mode="train"
#modelno=1
dataset="cmu"

create_cmd
cmd="${return_str} \\
-batchSize 4 \\
-modelVoxels models/t7/model_voxels_FVSV_cmu.t7 \\
-proj silhFVSV \\
"
run_cmd
```

generates and runs the following script: 
``` bash
cd ..
qlua main.lua \
-dirName segm15joints2Djoints3Dvoxels/rgb/gt_FVSV \
-input rgb \
-supervision segm15joints2Djoints3Dvoxels \
-datasetname cmu \
-batchSize 4 \
-modelVoxels models/t7/model_voxels_FVSV_cmu.t7 \
-proj silhFVSV \
```

This trains the final version of the model described in the paper, i.e., training end-to-end network with pre-trained subnetworks with multi-task losses and multi-view re-projection losses. If you manage to run this on the SURREAL dataset, the standard output should resemble the following:

```
Epoch: [1][1/2000] Time: 66.197, Err: 0.170      PCK: 87.50,    PixelAcc: 68.36,        IOU: 55.03,     RMSE: 0.00,     PE3Dvol: 33.39, IOUvox: 66.56,  IOUprojFV: 92.89,       IOUprojSV: 75.56,       IOUp
artvox: 0.00,    LR: 1e-03,      DataLoadingTime 192.286
Epoch: [1][2/2000] Time: 1.240, Err: 0.472      PCK: 87.50,    PixelAcc: 21.38,        IOU: 18.79,     RMSE: 0.00,     PE3Dvol: 44.63, IOUvox: 44.89,  IOUprojFV: 73.05,       IOUprojSV: 65.19,       IOUp
artvox: 0.00,    LR: 1e-03,      DataLoadingTime 0.237
Epoch: [1][3/2000] Time: 1.040, Err: 0.318      PCK: 65.00,    PixelAcc: 49.58,        IOU: 35.99,     RMSE: 0.00,     PE3Dvol: 52.92, IOUvox: 57.04,  IOUprojFV: 86.97,       IOUprojSV: 66.29,       IOUp
artvox: 0.00,    LR: 1e-03,      DataLoadingTime 0.570
Epoch: [1][4/2000] Time: 1.678, Err: 0.771       PCK: 50.00,    PixelAcc: 42.95,        IOU: 36.04,     RMSE: 0.00,     PE3Dvol: 99.04, IOUvox: 52.74,  IOUprojFV: 83.87,       IOUprojSV: 64.07,       IOUp
artvox: 0.00,    LR: 1e-03,      DataLoadingTime 0.101
```

2D pose (PCK), 2D body part segmentation (`PixelAcc`, `IOU`), depth (`RMSE`), 3D pose (`PE3Dvol`), voxel prediction (`IOUvox`), side-view and front-view re-projection (`IOUprojFV`, `IOUprojSV`) performances are reported at each iteration.

The final network is a result of a multi-stage training.
* SubNet1 - `model_segm_cmu.t7`. RGB -> **Segm**
  * obtained from [here](https://github.com/gulvarol/surreal) and the first two stacks are extracted
* SubNet2 - `model_joints2D.t7`. RGB -> **Joints2D**
  * trained on MPII with 8 stacks, and the first two stacks are extracted
* SubNet3 - `model_joints3D_cmu.t7`. RGB + Segm + Joints2D -> **Joints3D**
  * trained from scratch with 2 stacks using predicted segmentation (SubNet1) and 2D pose (SubNet2)
* SubNet4 - `model_voxels_cmu.t7`. RGB + Segm + Joints2D + Joints3D -> **Voxels**
  * trained from scratch with 2 stacks using predicted segmentation (SubNet1), 2D pose (SubNet2), and 3D pose (SubNet3)
* SubNet5 - `model_voxels_FVSV_cmu.t7`. RGB + Segm + Joints2D + Joints3D -> **Voxels + FV + SV**
  * pre-trained from SubNet4 with the additional losses on re-projection
* BodyNet - `model_bodynet_cmu.t7`. RGB -> **Segm + Joints2D + Joints3D + Voxels + FV + SV**
  * a combination of SubNet1, SubNet2, SubNet3, SubNet4, and SubNet5
  * fine-tuned end-to-end with multi-task losses

Note that the performance with 8 stacks is generally better, but we preferred to reduce the complexity with the cost of a little performance.

Above recipe is used for the SURREAL dataset. For the UP dataset, we first fine-tuned the SubNet1 `model_segm_UP.t7` (SubNet1_UP). Then, we fine-tuned SubNet3 `model_joints3D_UP.t7` (SubNet3_UP) using SubNet1_UP and SubNet2. Finally, we fine-tuned SubNet5 `model_voxels_FVSV_UP.t7` (SubNet5_UP) using SubNet1_UP, SubNet2, and SubNet3_UP. All these are fine-tuned end-to-end to obtain `model_bodynet_UP.t7`. The model used in the paper for experimenting with the manual segmentations is also provided `model_voxels_FVSV_UP_manualsegm.t7`.

### Part Voxels
We use the script `models/init_partvoxels.lua` to copy the last layer weights 7 times (6 body parts + 1 background) to initialize the part voxels model (`models/t7/init_partvoxels.t7`). After training this model without re-projection losses, we fine-tune it with re-projection loss. `model_partvoxels_cmu.t7` is the best model obtained. With end-to-end fine-tuning, we had divergence problems and did not put too much effort to make it work. Note that this model is preliminary and needs improvement.

### Misc
A few functionalities of the code are not used in the paper; however, still provided. These include training 3D pose and voxels networks using ground truth (GT) segmentation/2D pose/3D pose inputs, as well as mixing the predicted and GT inputs at each batch. This is achieved by setting the `mix` option to true. The results of only using predicted inputs are often comparable to using a mix, therefore we always used only predictions. Predictions are passed as input using the `applyHG` option, which is not very efficient.

## 3. Testing
Use the demo script to apply the provided models on sample images.
```
qlua demo/demo.lua
```
You can also use `demo/demo.m` Matlab script to produce visualizations.

## 4. Fitting SMPL model
Fitting scripts for SURREAL (`fitting/fit_surreal.py`) and UP (`fitting/fit_up.py`) datasets are provided with sample experiment outputs. The scripts use the optimization functions from `tools/smpl_utils.py`.

## Citation
If you use this code, please cite the following:

```
@INPROCEEDINGS{varol18_bodynet,
  title     = {{BodyNet}: Volumetric Inference of {3D} Human Body Shapes},
  author    = {Varol, G{\"u}l and Ceylan, Duygu and Russell, Bryan and Yang, Jimei and Yumer, Ersin and Laptev, Ivan and Schmid, Cordelia},
  booktitle = {ECCV},
  year      = {2018}
}
```

## Acknowledgements
The training code is an extension of the [SURREAL training code](https://github.com/gulvarol/surreal) which is largely built on the ImageNet training example [https://github.com/soumith/imagenet-multiGPU.torch](https://github.com/soumith/imagenet-multiGPU.torch) by [Soumith Chintala](https://github.com/soumith/), and [Stacked Hourglass Networks](https://github.com/umich-vl/pose-hg-train) by [Alejandro Newell](https://github.com/anewell).

The fitting code is an extension of the [SMPLify code](http://smplify.is.tue.mpg.de/).
