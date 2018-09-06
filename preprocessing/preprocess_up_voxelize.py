# Creates _info.mat files with the following information
#                    rt: [0 0 0]
#     joints2Ddeepercut: [3x14 double]
#                     f: 5000
#                  pose: [1x72 double]
#              cropinfo: '187 184 0 187 0 184'
#              joints2D: [25x2 double]
#                 shape: [1.6238 -0.4912 0.1992 -0.2878 0.2022 -0.1135 0.0085 -0.1162 -0.0530 0.0800]
#                     t: [0.0742 0.3018 56.4331]
#           datasetinfo: 'lsp 1002'
#                 trans: [0 0 0]
# This is added later to the _info.mat file
#              cropsegm: [1x1 struct]
#
# Creates _shape.mat files with the following information
#       voxelstranslate: [-0.8087 -0.9165 -0.6454]
#           voxelsscale: 1.3616
#         J_transformed: [24x3 double]
#                points: [6890x3 double]
#            voxelsdims: [123 128 125]
#                voxels: [123x125x128 logical]
#
# This is added later to the _shape.mat file
#            voxelsfill: [123x125x128 logical]
#
# Example usage:
#
# source activate smplenv
# export SMPL_PATH=smpl
# export BINVOX_PATH=tools/binvox/
# export BINVOX_PYTHON_PATH=tools/binvox/binvox-rw-py/
# python preprocess_up_voxelize.py

import numpy as np
import os
import pickle
import scipy.io as sio
from subprocess import call
import sys

curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(curr_dir, '../tools/'))
import smpl_utils

SMPL_PATH = os.getenv('SMPL_PATH', os.path.join(curr_dir, '../tools/smpl'))
BINVOX_PATH = os.getenv('BINVOX_PATH', os.path.join(curr_dir, '../tools/binvox/'))
BINVOX_PYTHON_PATH = os.getenv('BINVOX_PYTHON_PATH', os.path.join(curr_dir, '../tools/binvox/binvox-rw-py/'))
sys.path.append(SMPL_PATH)
sys.path.append(BINVOX_PYTHON_PATH)

import binvox_rw
from smpl_webuser.serialization import load_model

# Load SMPL model
model = load_model(os.path.join(SMPL_PATH, 'models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'))


def saveVoxelsBinvox(pose, shape, fileshape):
    if os.path.exists(fileshape):
        print('Already exists ' + fileshape)
    else:
        dict_shape = {}
        # Save gt points
        m = model.copy()
        m.betas[:] = shape
        m.pose[:] = pose
        dict_shape['points'] = m.r
        dict_shape['J_transformed'] = m.J_transformed.r
        # Write to .obj file
        obj_path = fileshape[:-10] + '.obj'
        smpl_utils.save_smpl_obj(obj_path, m, saveFaces=True)
        # Voxelize using binvox
        call([os.path.join(BINVOX_PATH, "binvox"), "-e", "-fit", "-d", "128", "%s" % obj_path])
        # Read the output of binvox
        binvox_path = obj_path[:-4] + '.binvox'
        with open(binvox_path, 'rb') as f:
            binvoxModel = binvox_rw.read_as_3d_array(f)
        # Remove intermediate files
        call(["rm", obj_path])
        call(["rm", binvox_path])
        # Save binvox results to mat
        dict_shape['voxels'] = binvoxModel.data
        dict_shape['voxelsdims'] = binvoxModel.dims
        dict_shape['voxelstranslate'] = binvoxModel.translate
        dict_shape['voxelsscale'] = binvoxModel.scale
        sio.savemat(fileshape, dict_shape, do_compression=True)
        print('Saved ' + fileshape)


def main():
    datapath = '/home/gvarol/data2/datasets/UP/'
    # Run for each setname: 'train' | 'val' | 'test'
    setname = 'val'

    targetpath = os.path.join(datapath, 'data', setname + '_up', 'dummy')
    if not os.path.exists(targetpath):
        os.makedirs(targetpath)

    with open(os.path.join(datapath, 'up-3d', setname + '.txt'), 'r') as f:
        testnames = f.readlines()
    testnames = [x.strip()[1:] for x in testnames]

    cnt = 1
    for filename in os.listdir(os.path.join(datapath, 'up-3d')):
        if filename.endswith("_image.png") and filename in testnames:
            try:
                cnt = cnt + 1
                filergb = os.path.join(datapath, 'up-3d', filename)
                fileroot = filergb[0:-10]
                filebody = fileroot + '_body.pkl'
                filecrop = fileroot + '_fit_crop_info.txt'  # height width y1 y2 x1 x2
                filedataset = fileroot + '_dataset_info.txt'  # lsp 17
                filej2d = fileroot + '_joints.npy'  # 2D joint predictions by deepcut 3x14 (x, y, confidence)
                fileinfo = os.path.join(targetpath, filename[0:-10] + '_info.mat')
                fileshape = os.path.join(targetpath, filename[0:-10] + '_shape.mat')

                with open(filebody, 'rb') as f:
                    bodyinfo = pickle.load(f)
                with open(filecrop, 'rb') as f:
                    cropinfo = f.readlines()
                with open(filedataset, 'rb') as f:
                    datasetinfo = f.readlines()
                with open(filej2d, 'rb') as f:
                    joints2Dpred = np.load(f)

                dict_info = {}
                dict_info['pose'] = bodyinfo['pose']  # 72
                dict_info['shape'] = bodyinfo['betas']  # 10
                dict_info['joints2D'] = bodyinfo['j2d']  # 25x2
                dict_info['trans'] = bodyinfo['trans']  # (0 0 0)
                dict_info['f'] = bodyinfo['f']  # (5000)
                dict_info['rt'] = bodyinfo['rt']  # (0 0 0)
                dict_info['t'] = bodyinfo['t']  # (x y z)
                dict_info['cropinfo'] = cropinfo
                dict_info['datasetinfo'] = datasetinfo
                dict_info['joints2Ddeepercut'] = joints2Dpred

                sio.savemat(fileinfo, dict_info, do_compression=True)
                saveVoxelsBinvox(dict_info['pose'], dict_info['shape'], fileshape)

            except:
                print("Oops")


if __name__ == '__main__':
    main()
