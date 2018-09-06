import matplotlib.pyplot as plt
import numpy as np
from scipy import misc
import scipy.io as sio
from skimage import measure
import sys
import os
from os.path import exists, join

curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(join(curr_dir, '../tools/'))
import smpl_utils
SMPL_PATH = os.getenv('SMPL_PATH', join(curr_dir, '../tools/smpl'))
sys.path.append(SMPL_PATH)
from smpl_webuser.serialization import load_model


class meshobj:
    def __init__(self, vertices, faces):
            self.r = vertices
            self.f = faces


def mkdir_safe(directory):
    if not exists(directory):
        try:
            os.makedirs(directory)
        except:  # FileExistsError:
            pass


def main():
    # <========= PARSE ARGUMENTS
    import argparse
    parser = argparse.ArgumentParser(description='Fit SMPL body to voxels predictions.')
    parser.add_argument('--testno', type=int, default=83,
                        help='ID of the tested image (0-based).')
    parser.add_argument('--thr', type=float, default=0.5,
                        help='Value to threshold the voxels.')
    parser.add_argument('--setname', default='lsp',
                        help='Test set name. Options: lsp | bmvc')
    args = parser.parse_args()
    testno = args.testno
    thr = args.thr
    setname = args.setname
    print('Options:')
    print('\tinput testno: %d' % testno)
    print('\tinput thr: %.1f' % thr)
    print('\tsetname: %s' % setname)
    # =========>

    # <========= LOAD SMPL MODEL
    m = load_model(join(SMPL_PATH, 'models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'))
    # Init upright t-pose
    initial_model = m.copy()
    initial_model.pose[0:3] = np.array((np.pi, 0, 0))
    # =========>

    # <========= DEFINITIONS
    voxelSize = 128.0
    datapath = 'sample_data/up/'
    obj_path = '%s/obj%0.1f/' % (datapath, thr)
    fig_path = '%s/fig%0.1f/' % (datapath, thr)
    mat_path = '%s/mat%0.1f/' % (datapath, thr)
    mkdir_safe(obj_path)
    mkdir_safe(fig_path)
    mkdir_safe(mat_path)

    with open('testnames/namesUP' + setname + '.txt', 'r') as f:
        names = f.readlines()
    # =========>

    # <========= LOAD files
    fileinfo = join(datapath, os.path.basename(names[testno][0:-12] + '_info.mat'))
    filergb = join(datapath, os.path.basename(names[testno][0:-12] + '_image.png'))
    rgb = misc.imread(filergb)
    fileoutput = join(datapath, 'outputs_%d.mat' % (testno + 1))
    filesegm = join(datapath, 'segm_%03d.mat' % (testno + 1))
    info = sio.loadmat(fileinfo)
    dict_segm = sio.loadmat(filesegm)
    dict_pred = sio.loadmat(fileoutput)
    # =========>

    # Set ground truth pose and shape
    m.pose[:] = info['pose']
    m.betas[:] = info['shape']

    # <========= NETWORK output
    voxels_gt = dict_pred['gt']  # gt voxels in 128-128-128
    voxels_pred = dict_pred['pred']  # gt voxels in 128-128-128
    voxels_gt = np.transpose(voxels_gt, (2, 1, 0))  # zyx -> xyz
    voxels_pred = np.transpose(voxels_pred, (2, 1, 0))
    voxels_gt = voxels_gt[:, :, ::-1]
    voxels_pred = voxels_pred[:, :, ::-1]
    # =========>

    # <========= BINVOX params from ground truth model m
    binvox_trans = np.min(m.r, axis=0)
    binvox_dim = np.max(m.r, axis=0) - binvox_trans
    binvox_scale = max(binvox_dim)
    # =========>

    # <========= SCALING params from segmentation
    segm = dict_segm['segm'].transpose()
    # Bbox in the segm image
    segmix = np.array(np.transpose(segm.nonzero()))
    segMaxX, segMaxY = np.max(segmix, axis=0)
    segMinX, segMinY = np.min(segmix, axis=0)
    # Scale is the longest segmentation bbox dimension
    segmScale = max((segMaxX - segMinX + 1) / voxelSize, (segMaxY - segMinY + 1) / voxelSize)
    voxels_gt_tight_shape = np.round(voxelSize * binvox_dim / binvox_scale)
    segmScale = np.round(segmScale * voxels_gt_tight_shape) / voxels_gt_tight_shape
    # =========>

    # <========= PADDING params from padded ground truth voxels
    voxels_gt_points = np.array(np.transpose((voxels_gt > 0.5).nonzero()))
    padding = np.min(voxels_gt_points, axis=0).astype(int)  # min x,y,z coordinates that have a filled voxel
    # ==========>

    # <========= RUN MARCHING CUBES ON THE PREDICTED VOXELS
    marching_vertices, triangles, normals, values = measure.marching_cubes_lewiner(voxels_pred, thr)
    # =========>

    # <========= Transform vertices to be at SMPL coordinates, remove padding and scale
    marching_vertices = ((marching_vertices - padding) / segmScale) / voxelSize * binvox_scale + binvox_trans
    # =========>

    filejoints3D = join(datapath, 'joints3D_%d.mat' % (testno + 1))
    dict_joints3D = sio.loadmat(filejoints3D)
    joints3D = dict_joints3D['pred']
    jix = np.array((5, 4, 3, 2, 1, 0, 6, 7, 8, 9, 15, 14, 13, 12, 11, 10))
    joints3D = joints3D[jix]

    if joints3D.shape[0] == 16 and all(v == 0 for v in joints3D[6, :]):
        joints3D = joints3D + m.J_transformed[0, :].r

    # <========= FIT SMPL BODY MODEL
    print('\n\n=> (1) Find theta that fits joints3D with beta=0\n\n')
    smpl_utils.optimize_on_joints3D(model=initial_model, joints3D=joints3D, viz=False)

    print('\n\n=> (2) Find trans that fits voxels and joints3D (optional)\n\n')
    trans_model = initial_model.copy()
    smpl_utils.iterative_optimize_on_vertices(
        model=trans_model,
        vertices=marching_vertices,
        joints3D=trans_model.J_transformed.r,
        vertices_prob=values,
        opt_cross=False,
        opt_trans=True,
        itr=15,
        viz=False)

    print('\n\n=> (3) Find theta and beta that fit voxels and joints3D\n\n')
    final_model = trans_model.copy()
    smpl_utils.iterative_optimize_on_vertices(
        model=final_model,
        vertices=marching_vertices,
        joints3D=final_model.J_transformed.r,
        vertices_prob=values,
        opt_cross=True,
        itr=3,
        viz=False)
    # =========>

    # <========= COMPUTE SURFACE ERROR
    surface_err = np.sqrt((np.power(final_model.r - m.r, 2).sum(axis=1))).mean() * 1000
    print('Surface error: %.2f' % surface_err)
    # =========>

    # <========= VISUALIZE
    plt.figure(figsize=(35, 10))
    plt.subplot(1, 4, 1)
    plt.imshow(np.fliplr(rgb))
    plt.title('RGB input')
    plt.subplot(1, 4, 2)
    smpl_utils.renderBody(m)
    plt.title('Ground truth')
    plt.subplot(1, 4, 3)
    smpl_utils.renderBody(initial_model)
    plt.title('Initial fit')
    plt.subplot(1, 4, 4)
    smpl_utils.renderBody(final_model)
    plt.title('Final fit')
    plt.savefig('%s/image_%03d.jpg' % (fig_path, (testno + 1)))
    # =========>

    # <========= WRITE MESH OBJ FILES
    smpl_utils.save_smpl_obj(join(obj_path, '%s_gt.obj' % (testno + 1)), m)
    smpl_utils.save_smpl_obj(join(obj_path, '%s_initial.obj' % (testno + 1)), initial_model)
    smpl_utils.save_smpl_obj(join(obj_path, '%s_final.obj' % (testno + 1)), final_model)
    smpl_utils.save_smpl_obj(join(obj_path, '%s_mcubes.obj' % (testno + 1)), meshobj(marching_vertices, triangles))
    # =========>

    # <========= WRITE SMPL PARAMS TO MAT FILES
    dict_params = {}
    dict_params['gt'] = {}
    dict_params['gt']['pose'] = m.pose.r
    dict_params['gt']['betas'] = m.betas.r
    dict_params['gt']['vertices'] = m.r
    dict_params['gt']['joints3D'] = m.J_transformed.r

    dict_params['initial'] = {}
    dict_params['initial']['pose'] = initial_model.pose.r
    dict_params['initial']['betas'] = initial_model.betas.r
    dict_params['initial']['vertices'] = initial_model.r
    dict_params['initial']['joints3D'] = initial_model.J_transformed.r

    dict_params['final'] = {}
    dict_params['final']['pose'] = final_model.pose.r
    dict_params['final']['betas'] = final_model.betas.r
    dict_params['final']['vertices'] = final_model.r
    dict_params['final']['joints3D'] = final_model.J_transformed.r

    dict_params['mcubes'] = {}
    dict_params['mcubes']['vertices'] = marching_vertices
    dict_params['mcubes']['triangles'] = triangles
    dict_params['mcubes']['values'] = values

    sio.savemat(join(mat_path, 'smpl_params_%s.mat' % (testno + 1)), dict_params)
    # =========>


if __name__ == '__main__':
    main()
