import argparse
import math
import numpy as np
import os
import pickle
import scipy.io as sio
from scipy import ndimage
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
devnull = open(os.devnull, 'w')

import binvox_rw
from smpl_webuser.serialization import load_model

# Load SMPL model
model_m = load_model(os.path.join(SMPL_PATH, 'models/basicModel_m_lbs_10_207_0_v1.0.0.pkl'))
model_f = load_model(os.path.join(SMPL_PATH, 'models/basicModel_f_lbs_10_207_0_v1.0.0.pkl'))


def saveVoxelsBinvox(fileinfo):
    filevoxels = fileinfo[:-9] + '_voxels.mat'

    if os.path.exists(filevoxels):
        print('Already exists.')
    else:
        info = sio.loadmat(fileinfo)
        # Get the default model according to the gender
        if info['gender'][0] == 0:
            m = model_f
        elif info['gender'][0] == 1:
            m = model_m
        # SMPL pose parameters for all frames
        pose = info['pose']
        # SMPL shape parameter is constant throughout the same clip
        shape = info['shape'][:, 0]
        # body rotation in euler angles
        zrot = info['zrot']
        zrot = zrot[0][0]
        RzBody = np.array(((math.cos(zrot), -math.sin(zrot), 0), (math.sin(zrot), math.cos(zrot), 0), (0, 0, 1)))

        T = pose.shape[1]
        dict_voxels = {}
        for t in range(T):
            print('FRAME %d' % t)
            # Rotate the body by zrot
            p = pose[:, t]
            p[0:3] = smpl_utils.rotateBody(RzBody, p[0:3])
            # Set body shape (beta)
            m.betas[:] = shape
            # Set body pose (theta)
            m.pose[:] = p

            # Write to an .obj file
            obj_path = fileinfo[:-9] + '_%d.obj' % t
            smpl_utils.save_smpl_obj(obj_path, m, saveFaces=True, verbose=False)

            call([os.path.join(BINVOX_PATH, "binvox"), "-e", "-fit", "-d", "128", "%s" % obj_path],
                 stdout=devnull, stderr=devnull)

            binvox_path = obj_path[:-4] + '.binvox'
            with open(binvox_path, 'rb') as f:
                binvoxModel = binvox_rw.read_as_3d_array(f)

            call(["rm", binvox_path])
            call(["rm", obj_path])
            dict_voxels['voxels_%d' % (t + 1)] = binvoxModel.data

        print(filevoxels)
        sio.savemat(filevoxels, dict_voxels, do_compression=True)


def savePartVoxelsBinvox(fileinfo):
    save_surface = False

    filevoxels = fileinfo[:-9] + '_partvoxels.mat'
    dirname = os.path.dirname(filevoxels)

    def mkdir_safe(directory):
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
            except FileExistsError:
                pass

    if os.path.exists(filevoxels):
        print('Already exists.')
        exit()
    else:
        mkdir_safe(dirname)

        info = sio.loadmat(fileinfo)

        # Get the neutral model according to the gender
        if info['gender'][0] == 0:
            m = model_f
        elif info['gender'][0] == 1:
            m = model_m

        # SMPL pose parameters for all frames
        pose = info['pose']
        # SMPL shape parameter is constant throughout the same clip
        shape = info['shape'][:, 0]
        zrot = info['zrot']
        # body rotation in euler angles
        zrot = zrot[0][0]
        RzBody = np.array(((math.cos(zrot), -math.sin(zrot), 0), (math.sin(zrot), math.cos(zrot), 0), (0, 0, 1)))

        # SEGMENTATION
        with open("parts/segm_per_v_overlap.pkl", "r") as f:
            segm_v = pickle.load(f)  # 0-based

        sorted_parts = ['hips', 'leftUpLeg', 'rightUpLeg', 'spine', 'leftLeg', 'rightLeg',
                        'spine1', 'leftFoot', 'rightFoot', 'spine2', 'leftToeBase', 'rightToeBase',
                        'neck', 'leftShoulder', 'rightShoulder', 'head', 'leftArm', 'rightArm',
                        'leftForeArm', 'rightForeArm', 'leftHand', 'rightHand', 'leftHandIndex1', 'rightHandIndex1']

        # Convert into 6 parts of head, torso, left/right leg, left/right arm
        segmix = np.array((2, 6, 5, 2, 6, 5, 2, 6, 5, 2, 6, 5, 2, 2, 2, 1, 4, 3, 4, 3, 4, 3, 4, 3)) - 1
        s = len(np.unique(segmix))

        S = {}
        for partno in range(s):
            S[partno] = []

        for partno in range(24):
            indices = segm_v[sorted_parts[partno]]
            six = segmix[partno]
            S[six] = S[six] + indices

        T = pose.shape[1]
        dict_voxels = {}
        # For each frame
        for t in range(T):
            print('FRAME %d' % t)
            # Rotate the body by zrot
            p = pose[:, t]
            p[0:3] = smpl_utils.rotateBody(RzBody, p[0:3])
            # Set body shape (beta)
            m.betas[:] = shape
            # Set body pose (theta)
            m.pose[:] = p

            # BINVOX params from ground truth model m, min vertex coordinates is the translation from binvox
            binvox_min = np.min(m.r, axis=0)
            binvox_max = np.max(m.r, axis=0)

            if save_surface:
                partvoxels = np.zeros((128, 128, 128), dtype=np.int8)

            partvoxelsfill = np.zeros((128, 128, 128), dtype=np.int8)

            # Iterate on the torso (index 1) as last to assign overlapping voxels to torso
            partslist = [0, 2, 3, 4, 5, 1]

            # For each body part
            for partno in partslist:

                # Create the part obj
                # Lines about the faces are a bit redundant
                faces_subset = m.f[np.all(np.isin(m.f, S[partno]), axis=1)]
                I, newVertIndices = np.unique(faces_subset, return_inverse=True)
                faces = np.reshape(newVertIndices, faces_subset.shape)
                vertices = m.r[I, :]

                if save_surface:
                    # Write to an .obj file
                    obj_path = fileinfo[:-9] + '_%d_part%d.obj' % (t, partno)
                    smpl_utils.save_obj(obj_path, vertices, faces=faces, verbose=False)
                    # Without -fit option this time. (min_x min_y min_z max_x max_y max_z)
                    call([os.path.join(BINVOX_PATH, "binvox"), "-e", "-d", "128", "%s" % obj_path, "-bb",
                          "%f" % binvox_min[0],
                          "%f" % binvox_min[1],
                          "%f" % binvox_min[2],
                          "%f" % binvox_max[0],
                          "%f" % binvox_max[1],
                          "%f" % binvox_max[2]],
                         stdout=devnull, stderr=devnull)

                    binvox_path = obj_path[:-4] + '.binvox'
                    with open(binvox_path, 'rb') as f:
                        binvoxModel = binvox_rw.read_as_3d_array(f)
                    call(["rm", binvox_path])
                    call(["rm", obj_path])
                    vsurface = binvoxModel.data
                    partvoxels[vsurface == 1] = partno + 1

                # Write to an .obj file
                obj_path_fill = fileinfo[:-9] + '_%d_part%d_fill.obj' % (t, partno)
                with open(obj_path_fill, 'w') as fp:
                    for v in vertices:
                        fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
                    # Faces are 1-based, not 0-based in obj files
                    for f in faces + 1:
                        fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
                    # Add extra faces to close the hole
                    holes_path = 'parts/holes/part%d.txt' % (partno + 1)
                    with open(holes_path, 'r') as fh:
                        for line in fh:
                            fp.write(line)

                # Run binvox (min_x min_y min_z max_x max_y max_z)
                call([os.path.join(BINVOX_PATH, "binvox"), "-e", "-d", "128", "%s" % obj_path_fill, "-bb",
                      "%f" % binvox_min[0],
                      "%f" % binvox_min[1],
                      "%f" % binvox_min[2],
                      "%f" % binvox_max[0],
                      "%f" % binvox_max[1],
                      "%f" % binvox_max[2]],
                     stdout=devnull, stderr=devnull)

                binvox_path_fill = obj_path_fill[:-4] + '.binvox'
                with open(binvox_path_fill, 'rb') as f:
                    binvoxModel_fill = binvox_rw.read_as_3d_array(f)
                call(["rm", binvox_path_fill])
                call(["rm", obj_path_fill])
                vfill = ndimage.binary_fill_holes(binvoxModel_fill.data)

                partvoxelsfill[vfill == 1] = partno + 1

            xyz = np.nonzero(partvoxelsfill)
            minx = min(xyz[0])
            miny = min(xyz[1])
            minz = min(xyz[2])
            maxx = max(xyz[0])
            maxy = max(xyz[1])
            maxz = max(xyz[2])

            # e.g. size of tightpartvoxels: (46, 128, 42) and partvoxels: (128, 128, 128)
            if save_surface:
                tightpartvoxels = partvoxels[minx:maxx + 1, miny:maxy + 1, minz:maxz + 1]

            tightpartvoxelsfill = partvoxelsfill[minx:maxx + 1, miny:maxy + 1, minz:maxz + 1]

            # dims/translate/scale are common for all voxels since given a fix bbox
            # dict_voxels['partvoxels_dims_%d' % (t+1)] = binvoxModel_fill.dims # this is always [128,128,128], no need to save
            dict_voxels['partvoxels_translate_%d' % (t + 1)] = binvoxModel_fill.translate
            dict_voxels['partvoxels_scale_%d' % (t + 1)] = binvoxModel_fill.scale
            dict_voxels['partvoxelsfill_%d' % (t + 1)] = tightpartvoxelsfill
            if save_surface:
                dict_voxels['partvoxels_%d' % (t + 1)] = tightpartvoxels

        print(filevoxels)
        sio.savemat(filevoxels, dict_voxels, do_compression=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        default='sample_data/surreal/03_01_c0008_info.mat',
        help='input _info.mat path from SURREAL dataset')
    parser.add_argument(
        '--parts',
        action='store_true',
        help='whether to voxelize parts (default: False, voxelizes body)')
    opts = parser.parse_args()

    if opts.parts:
        print('Voxelizing parts: %s' % opts.input)
        savePartVoxelsBinvox(opts.input)
    else:
        print('Voxelizing: %s' % opts.input)
        saveVoxelsBinvox(opts.input)


if __name__ == '__main__':
    main()
