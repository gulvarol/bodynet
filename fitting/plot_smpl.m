addpath('../demo/matlab_utils'); % fig_smpl
load('sample_data/surreal/mat0.5/smpl_params_233.mat', 'final');
load('smpl_faces.mat', 'smpl_faces');
fig_smpl(final.vertices, 'sample_data/surreal/233', [0 0], smpl_faces+1);