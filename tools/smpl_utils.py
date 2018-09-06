import chumpy
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy
import sys
from time import time

import rotations

# SMPLify (http://smplify.is.tue.mpg.de/)
curr_dir = os.path.dirname(os.path.abspath(__file__))
SMPLIFY_PATH = os.getenv('SMPLIFY_PATH', os.path.join(curr_dir, 'smplify_public/code/'))
sys.path.append(SMPLIFY_PATH)
from lib.max_mixture_prior import MaxMixtureCompletePrior
from lib.robustifiers import GMOf


def optimize_on_joints3D(model,
                         joints3D,
                         opt_shape=False,
                         viz=True):
    """Fit the model to the given set of 3D joints
    :param model: initial SMPL model ===> is modified after optimization
    :param joints3D: 3D joint locations [16 x 3]
    :param opt_shape: boolean, if True optimizes for shape parameter betas
    :param viz: boolean, if True enables visualization during optimization
    """
    t0 = time()
    if joints3D.shape[0] == 16:
        obj_joints3D = lambda w, sigma: (w * GMOf((joints3D - model.J_transformed[get_indices_16()]), sigma))
    elif joints3D.shape[0] == 24:
        obj_joints3D = lambda w, sigma: (w * GMOf((joints3D - model.J_transformed), sigma))
    else:
        raise('How many joints?')

    # Create the pose prior (GMM over CMU)
    prior = MaxMixtureCompletePrior(n_gaussians=8).get_gmm_prior()
    pprior = lambda w: w * prior(model.pose)
    # joint angles pose prior, defined over a subset of pose parameters:
    # 55: left elbow,  90deg bend at -np.pi/2
    # 58: right elbow, 90deg bend at np.pi/2
    # 12: left knee,   90deg bend at np.pi/2
    # 15: right knee,  90deg bend at np.pi/2
    my_exp = lambda x: 10 * chumpy.exp(x)
    obj_angle = lambda w: w * chumpy.concatenate([my_exp(model.pose[55]),
                                                 my_exp(-model.pose[58]),
                                                 my_exp(-model.pose[12]),
                                                 my_exp(-model.pose[15])])
    # Visualization at optimization step
    if viz:
        def on_step(_):
            """Draw a visualization."""
            plt.figure(1, figsize=(5, 5))
            renderBody(model)
            plt.draw()
            plt.pause(1e-3)
    else:
        on_step = None

    # weight configuration (pose and shape: original values as in SMPLify)
    # the first list contains the weights for the pose prior,
    # the second list contains the weights for the shape prior
    opt_weights = zip([4.04 * 1e2, 4.04 * 1e2, 57.4, 4.78],
                      [1e2, 5 * 1e1, 1e1, .5 * 1e1])

    print('Initial: error(joints3D) = %.2f' % (obj_joints3D(100, 100).r**2).sum())

    # run the optimization in 4 stages, progressively decreasing the
    # weights for the priors
    for stage, (wpose, wbetas) in enumerate(opt_weights):
        objs = {}
        objs['joints3D'] = obj_joints3D(100., 100)
        objs['pose'] = pprior(wpose)
        objs['pose_exp'] = obj_angle(0.317 * wpose)
        if opt_shape:
            objs['betas'] = wbetas * model.betas

            chumpy.minimize(
                objs,
                x0=[model.pose, model.betas],
                method='dogleg',
                callback=on_step,
                options={'maxiter': 1000,
                         'e_3': .0001,
                         'disp': 0})
        else:
            chumpy.minimize(
                objs,
                x0=[model.pose],
                method='dogleg',
                callback=on_step,
                options={'maxiter': 1000,
                         'e_3': .0001,
                         'disp': 0})
        print('Stage %d: error(joints3D) = %.2f' % (stage, (objs['joints3D'].r**2).sum()))
    print('\nElapsed theta fitting (%d joints): %.2f sec.' % (joints3D.shape[0], (time() - t0)))


def iterative_optimize_on_vertices(model,
                                   vertices,
                                   joints3D,
                                   vertices_prob=np.zeros(1),
                                   opt_cross=True,
                                   opt_trans=False,
                                   itr=5,
                                   viz=True):
    """Fit the model (iteratively) to the given set of 3D vertices + 3D joints
    :param model: initial SMPL model ===> is modified after optimization
    :param vertices: 3D vertex locations to fit [num_vertices x 3]
    :param joints3D: 3D joint locations to fit [16 x 3]
    :param vertices_prob: an optional confidence/probability score for each point
    :param opt_trans: boolean, if True optimizes only translation
    :param itr: how many iterations of correspondence computation/fitting
    """
    t0 = time()
    # kdtree for vertices to make closest point queries
    pt_tree = scipy.spatial.cKDTree(vertices, leafsize=15)
    print('==> Will run for %d iterations to fit to %d vertices:' % (itr, vertices.shape[0]))
    for i in range(itr):
        print('\n===> Iteration %d' % i)
        smpl_tree = scipy.spatial.cKDTree(model.r, leafsize=15)

        vertices_corr = np.zeros((model.r.shape[0], 3))
        weights_corr = np.zeros((model.r.shape[0]))
        count = 0
        for v in model.r:
            dd, ii = pt_tree.query(v, k=1)
            vertices_corr[count, :] = vertices[ii, :]
            if(vertices_prob.shape[0] == 1):
                weights_corr[count] = 1.0
            else:
                weights_corr[count] = vertices_prob[ii]
            count = count + 1

        if opt_cross:
            print('====> Correspondence from both directions.')
            weights_cross_corr = np.zeros(vertices.shape[0])
            indices_cross_corr = np.zeros(vertices.shape[0])
            count = 0
            for j in range(vertices.shape[0]):
                pt = [vertices[j][0], vertices[j][1], vertices[j][2]]
                dd, ii = smpl_tree.query(pt, k=1)
                indices_cross_corr[count] = ii
                if(vertices_prob.shape[0] == 1):
                    weights_cross_corr[count] = 1.0
                else:
                    weights_cross_corr[count] = vertices_prob[j]
                count = count + 1
            optimize_on_vertices(
                model=model,
                vertices=vertices_corr,
                joints3D=joints3D,
                weights_corr=weights_corr,
                vertices_cross_corr=vertices,
                indices_cross_corr=indices_cross_corr,
                weights_cross_corr=weights_cross_corr,
                opt_trans=opt_trans,
                viz=viz)
        else:
            print('====> Correspondence from one direction.')
            optimize_on_vertices(
                model=model,
                vertices=vertices_corr,
                joints3D=joints3D,
                weights_corr=weights_corr,
                opt_trans=opt_trans,
                viz=viz)

    print('\nElapsed beta & theta fitting (%d vertices)(%d joints): %.2f sec.'
          % (vertices.shape[0], joints3D.shape[0], (time() - t0)))


def optimize_on_vertices(model,
                         vertices,
                         joints3D=np.zeros(1),
                         weights_corr=np.zeros(1),
                         vertices_cross_corr=np.zeros(1),
                         indices_cross_corr=np.zeros(1),
                         weights_cross_corr=np.zeros(1),
                         opt_trans=False,
                         viz=True):
    """Fit the model to the given set of 3D vertices and 3D joints
    :param model: initial SMPL model ===> is modified after optimization
    :param vertices: 3D vertex locations to fit [num_vertices x 3]
    :param joints3D: 3D joint locations to fit [24 x 3]
    :param vertices_cross_corr, indices_cross_corr, weights_cross_corr:
    :for each point in vertices_cross_corr, we have the index of its corresponding smpl vertex and the weight
    :for this correspondence
    :param opt_trans: boolean, if True optimizes only translation
    :param viz: boolean, if True enables visualization during optimization
    """
    t0 = time()
    # Optimization term on the joints3D distance
    if joints3D.shape[0] > 1:
        if joints3D.shape[0] == 16:
            obj_joints3d = lambda w, sigma: (w * GMOf((joints3D - model.J_transformed[get_indices_16()]), sigma))
        elif joints3D.shape[0] == 24:
            obj_joints3d = lambda w, sigma: (w * GMOf((joints3D - model.J_transformed), sigma))
        else:
            raise('How many joints?')

    # data term: distance between observed and estimated points in 3D
    if(weights_corr.shape[0] == 1):
        weights_corr = np.ones((vertices.shape[0]))

    obj_vertices = lambda w, sigma: (w * GMOf(((vertices.T * weights_corr)
                                     - (model.T * weights_corr)).T, sigma))

    if(vertices_cross_corr.shape[0] > 1):
        smplV = model[indices_cross_corr.astype(int), :]
        obj_vertices_cross = lambda w, sigma: (w * GMOf(((vertices_cross_corr.T * weights_cross_corr)
                                               - (smplV.T * weights_cross_corr)).T, sigma))
    # Create the pose prior (GMM over CMU)
    prior = MaxMixtureCompletePrior(n_gaussians=8).get_gmm_prior()
    pprior = lambda w: w * prior(model.pose)
    # joint angles pose prior, defined over a subset of pose parameters:
    # 55: left elbow,  90deg bend at -np.pi/2
    # 58: right elbow, 90deg bend at np.pi/2
    # 12: left knee,   90deg bend at np.pi/2
    # 15: right knee,  90deg bend at np.pi/2
    my_exp = lambda x: 10 * chumpy.exp(x)
    obj_angle = lambda w: w * chumpy.concatenate(
        [my_exp(model.pose[55]),
         my_exp(-model.pose[58]),
         my_exp(-model.pose[12]),
         my_exp(-model.pose[15])])
    # Visualization at optimization step
    if viz:
        def on_step(_):
            """Draw a visualization."""
            plt.figure(1, figsize=(5, 5))
            renderBody(model)
            plt.draw()
            plt.pause(1e-3)
    else:
        on_step = None

    # weight configuration (pose and shape: original values as in SMPLify)
    # the first list contains the weights for the pose prior,
    # the second list contains the weights for the shape prior
    # the third list contains the weights for the joints3D loss
    opt_weights = zip([4.04 * 1e2, 4.04 * 1e2, 57.4, 4.78],
                      [1e2, 5 * 1e1, 1e1, .5 * 1e1],
                      [5, 5, 5, 5])
    print('Initial:')
    print('\terror(vertices) = %.2f' % (obj_vertices(100, 100).r**2).sum())
    if(joints3D.shape[0] > 1):
        print('\terror(joints3d) = %.2f' % (obj_joints3d(100, 100).r**2).sum())
    if(vertices_cross_corr.shape[0] > 1):
        print('\terror(vertices_cross) = %.2f' % (obj_vertices_cross(100, 100).r**2).sum())

    # run the optimization in 4 stages, progressively decreasing the
    # weights for the priors 
    for stage, (wpose, wbetas, wjoints3D) in enumerate(opt_weights):
        print('Stage %d' % stage)
        objs = {}
        if(joints3D.shape[0] > 1):
            objs['joints3D'] = wjoints3D * obj_joints3d(100., 100)
        objs['vertices'] = obj_vertices(100., 100)
        if(vertices_cross_corr.shape[0] > 1):
            objs['vertices_cross'] = obj_vertices_cross(100., 100)
        objs['pose'] = pprior(wpose)
        objs['pose_exp'] = obj_angle(0.317 * wpose)
        objs['betas'] = wbetas * model.betas

        if opt_trans:
            chumpy.minimize(
                objs,
                x0=[model.trans],
                method='dogleg',
                callback=on_step,
                options={'maxiter': 1000,
                         'e_3': .0001,
                         'disp': 0})
        else:
            chumpy.minimize(
                objs,
                x0=[model.pose, model.betas],
                method='dogleg',
                callback=on_step,
                options={'maxiter': 1000,
                         'e_3': .0001,
                         'disp': 0})
        print('\terror(vertices) = %.2f' % (objs['vertices'].r**2).sum())
        if(joints3D.shape[0] > 1):
            print('\terror(joints3D) = %.2f' % (objs['joints3D'].r**2).sum())
        if(vertices_cross_corr.shape[0] > 1):
            print('\terror(vertices_cross) = %.2f' % (objs['vertices_cross'].r**2).sum())
    print('Elapsed iteration %.2f sec.' % (time() - t0))


def renderBody(m):
    from opendr.camera import ProjectPoints
    from opendr.renderer import ColoredRenderer
    from opendr.lighting import LambertianPointLight
    # Create OpenDR renderer
    rn = ColoredRenderer()
    # Assign attributes to renderer
    w, h = (640, 480)
    rn.camera = ProjectPoints(v=m,
                              rt=np.zeros(3),
                              t=np.array([0, 0, 2.]),
                              f=np.array([w, w]) / 2.,
                              c=np.array([w, h]) / 2.,
                              k=np.zeros(5))
    rn.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
    rn.set(v=m, f=m.f, bgcolor=np.zeros(3))
    # Construct point light source
    rn.vc = LambertianPointLight(
        f=m.f,
        v=rn.v,
        num_verts=len(m),
        light_pos=np.array([-1000, -1000, -2000]),
        vc=np.ones_like(m) * .9,
        light_color=np.array([1., 1., 1.]))
    plt.ion()
    plt.imshow(np.fliplr(rn.r))  # FLIPPED!
    plt.show()
    plt.xticks([])
    plt.yticks([])


def rotateBody(RzBody, pelvisRotVec):
    Rpelvis = rotations.rotvec2rotmat(pelvisRotVec)
    globRotMat = np.dot(RzBody, Rpelvis)
    R90 = rotations.euler2rotmat(np.array((np.pi / 2, np.pi / 2, 0)))
    globRotVec = rotations.rotmat2rotvec(np.dot(R90, globRotMat))
    return globRotVec


def save_smpl_obj(outmesh_path, m, saveFaces=True, verbose=True):
    with open(outmesh_path, 'w') as fp:
        for v in m.r:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        if saveFaces:
            for f in m.f + 1:  # Faces are 1-based, not 0-based in obj files
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
    if verbose:
        print('Output mesh saved to: %s\n' % outmesh_path)


def save_obj(outmesh_path, vertices, faces=[], verbose=True):
    with open(outmesh_path, 'w') as fp:
        for v in range(vertices.shape[0]):  # vertices
            fp.write('v %f %f %f\n' % (vertices[v][0], vertices[v][1], vertices[v][2]))
        if faces:
            # Faces are 1-based, not 0-based in obj files
            for f in faces + 1:
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
    if verbose:
        print('Output mesh saved to: %s\n' % outmesh_path)


def get_indices_16():
    return np.array([8, 5, 2, 3, 6, 9, 1, 7, 13, 16, 21, 19, 17, 18, 20, 22]) - 1
