clear; clc; close all;
%% Load: rgb, segm, joints2D, joints3D, voxels
fn = 'sample_data/up/input.png';
load([fn '.mat']);

%%
vix = [3 1 2];
jix = [7, 3, 4, 2, 5, 1, 6, 9, 10, 13, 14, 12, 15, 11, 16];
addpath('matlab_utils');
% From mathworks.com/matlabcentral/fileexchange/26710-smooth-triangulated-mesh
addpath('smoothpatch_version1b');
%addpath('export_fig'); % if you don't have this replace with saveas func

figure(1); set(gcf, 'Position', [1023 523 442 347]);
figure(2); set(gcf, 'Position', [132 95 820 805]);
vws = [0 0]; %; 45 ,20];

%% 2D figs
figure(1);

% (1) RGB
rgb = uint8(255*permute(rgb, [2 3 1]));
imshow(rgb);
imwrite(rgb, [fn '_rgb.png']);

% (2) 2D Pose + Segm
swap_segm = [1 2 3 7 8 9 4 5 6 13 14 15 10 11 12];
swapped_segm = segm;
for p = 1:15
    swapped_segm(segm == swap_segm(p)) = p;
end
segm = swapped_segm;
fig_2D(joints2D(jix, :)', segm, fn);

%% 3D figs
figure(2);

% (3) 3D Pose
joints3D(:, [1 2]) = -joints3D(:, [1 2]);
joints3D = joints3D(:, vix);
fig_joints3D(joints3D(jix, :)', fn, vws);

% (4) Voxels
voxels = permute(voxels, vix);
voxels = voxels(:, end:-1:1, end:-1:1);
fig_mesh(voxels, true, double(imresize(rgb, [128 128]))./255, fn, vws);

% (5) Part voxels
logm = 1 ./ (1 + exp(-partvoxels));
softm = bsxfun(@rdivide, logm,  sum(logm, 1));

[partprob, partclass] = max(softm, [], 1);
partprob = squeeze(partprob);
partclass = squeeze(partclass);

partclass = permute(partclass, vix);
partclass = partclass(:, end:-1:1, end:-1:1);
fig_parts(partclass, fn, vws);

if false
    % (6) SMPL
    fig_smpl(smpl.gt_vertices, [fn 'gt_'], vws, obj.f.v);
    fig_smpl(smpl.initial_vertices, [fn 'pred1_'], vws, obj.f.v);
    fig_smpl(smpl.final_vertices, [fn 'pred2_'], vws, obj.f.v);
end
