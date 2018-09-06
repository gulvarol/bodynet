clc; clear; close all;

% Rootpath of the dataset
rt = '~/datasets/UP/';

%% Read .txt files to make a list of train/test split names

% UP official train split
fid = fopen([rt 'up-3d/train.txt'], 'r');
train_up = textscan(fid, '%s');
train_up = train_up{1};
train_up = cellfun(@(x)(x(2:6)), train_up, 'UniformOutput', false);
fclose(fid);

% UP official train+val split
fid = fopen([rt 'up-3d/trainval.txt'], 'r');
trainval_up = textscan(fid, '%s');
trainval_up = trainval_up{1};
trainval_up = cellfun(@(x)(x(2:6)), trainval_up, 'UniformOutput', false);
fclose(fid);

% UP official val split
fid = fopen([rt 'up-3d/val.txt'], 'r');
val_up = textscan(fid, '%s');
val_up = val_up{1};
val_up = cellfun(@(x)(x(2:6)), val_up, 'UniformOutput', false);
fclose(fid);

% UP official test split
fid = fopen([rt 'up-3d/test.txt'], 'r');
test_up = textscan(fid, '%s');
test_up = test_up{1};
test_up = cellfun(@(x)(x(2:6)), test_up, 'UniformOutput', false);
fclose(fid);

% BMVC train split
setname = 'train';
fid = fopen([rt 'bmvc_subset/' setname 'list'], 'r');
train_bmvc = textscan(fid, '%s', 'Delimiter', ',');
train_bmvc = cellfun(@(x)(x(2:end-1)), train_bmvc{1, :}, 'UniformOutput', false);
train_bmvc{1} = train_bmvc{1}(2:end);
train_bmvc{end} = train_bmvc{end}(1:end-1);
fclose(fid);

% BMVC test split
setname = 'test';
fid = fopen([rt 'bmvc_subset/' setname 'list'], 'r');
test_bmvc = textscan(fid, '%s', 'Delimiter', ',');
test_bmvc = cellfun(@(x)(x(2:end-1)), test_bmvc{1, :}, 'UniformOutput', false);
test_bmvc{1} = test_bmvc{1}(2:end);
test_bmvc{end} = test_bmvc{end}(1:end-1);
fclose(fid);

% Make sure train/test split of bmvc is a subset of the official split
assert(all(ismember(train_bmvc, trainval_up)));
assert(all(ismember(test_bmvc, test_up)));

%% Save _segm.mat info for (segm1 and segm7 fields)

segmpath = [rt 'upi-s1h/'];
all_sets = {'train_up', 'val_up', 'test_up'};
for j = 1:length(all_sets);
    setstring = all_sets{j};
    setname = eval(setstring);
    for i = 1:length(setname) % (first 523 lsp)
        nm = setname{i};
        fid = fopen([rt 'up-3d/' nm '_dataset_info.txt']);
        dinfo = textscan(fid, '%s');
        fclose(fid);
        % Copy the image file
        system(['cp ' rt 'up-3d/' nm '_image.png ' rt 'data/' setstring '/dummy/' nm '_image.png']);
        % Create the segm file
        segmmat = [rt 'data/' setstring '/dummy/' nm '_segm.mat'];
        idno = str2num(dinfo{1}{2});
        if(strcmp(dinfo{1}{1}, 'lsp'))
            % Segm1
            segm1 = imread(sprintf('%s/lsp/im%04d_segmentation.png', segmpath, idno));
            segm1(segm1 == 255) = 1;
            % Segm7
            segm7 = imread(sprintf('%s/lsp/im%04d_part_segmentation.png', segmpath, idno));
            segm7(segm7 == 255) = 0;
            save(segmmat, 'segm1', 'segm7'); 
        elseif(strcmp(dinfo{1}{1}, 'lspext'))
            % Segm1
            filename = sprintf('%s/lsp_extended/im%05d_segmentation.png', segmpath, idno);
            if(~exist(filename))
                disp(i);
                continue;
            end
            segm1 = imread(filename);
            segm1 = segm1(:, :, 1);
            segm1(segm1 == 255) = 1;
            save(segmmat, 'segm1'); 
        elseif(strcmp(dinfo{1}{1}, 'mpii'))
            % Segm1
            filename = sprintf('%s/mpii/images/%05d_segmentation.png', segmpath, idno);
            if(~exist(filename))
                disp(i);
                continue;
            end
            segm1 = imread(filename);
            segm1 = segm1(:, :, 1);
            segm1(segm1 == 255) = 1;
            save(segmmat, 'segm1'); 
        end
        %subplot(1, 2, 1); imshow(imread([rt 'up-3d/' nm '_image.png']));
        %subplot(1, 2, 2); imagesc(segm1); axis equal; drawnow;
    end
end

% Note: upi-s1h/lsp_extended/im00050_segmentation.png" does not exist!

%% Add segm31 field in the _segm.mat files

% Structure of the original dataset:
% up-s31/s31/
%   trainval_31_500_pkg_dorder.txt (7126)
%   train_31_500_pkg_dorder.txt (5703)
%   val_31_500_pkg_dorder.txt (1423) [e.g. /31/500/pkg_dorder/08130_image.png /31/500/pkg_dorder/08130_ann.png 1.902011]
%   test_31_500_pkg_dorder.txt (1389)
%     05d_ann.png
%     05d_ann_vis.png
%     05d_image.png

% Run this cell for each setname: 'train', 'val', 'test'
setname = 'train';
fid = fopen(fullfile(rt, sprintf('up-s31/%s_31_500_pkg_dorder.txt', setname)));
annot = textscan(fid, '%s %s %f');
N = length(annot{1});
cropsegmcell = cell(N, 1);
segm31cell = cell(N, 1);

% Compute segm31 and cropsegm for all images in the set
parfor i = 1:N
    [~, bn] = fileparts(annot{1}{i});
    nm = bn(1:5);
    
    infofile = fullfile(rt, ['data/' setname '_up/dummy'],   [nm '_info.mat']);
    
    img = imread(fullfile(rt, ['data/' setname '_up/dummy'], [nm '_image.png']));
    segm = imread(fullfile(rt, 'up-s31', [nm '_ann.png']));
    
    cropsegmcell{i}.scale = annot{3}(i);
    
    a = load(infofile);
    imgsize = round(cropsegmcell{i}.scale * [size(img, 1), size(img, 2)]);
    person_center = cropsegmcell{i}.scale * mean(a.joints2Ddeepercut(1:2, a.joints2Ddeepercut(3, :) == 1), 2);
    
    % Uses the official code to avoid mistakes
    [~, pyout] = system(sprintf('python up_crop.py %d %d %d %d', imgsize(1), imgsize(2), floor(person_center(1)), floor(person_center(2))));
    pyout = strsplit(pyout, ' ');
    cropsegmcell{i}.crop_y = [str2num(pyout{1}) str2num(pyout{2})];
    cropsegmcell{i}.crop_x = [str2num(pyout{3}) str2num(pyout{4})];
    cy = round(cropsegmcell{i}.crop_y / cropsegmcell{i}.scale);
    cx = round(cropsegmcell{i}.crop_x / cropsegmcell{i}.scale);
    
    img_scale = imresize(segm, 1/cropsegmcell{i}.scale, 'nearest');
    
    if(cy(2) - cy(1) ~= size(img_scale, 1))
        cy(2) = cy(1) + size(img_scale, 1);
    end
    if(cx(2) - cx(1) ~= size(img_scale, 2))
        cx(2) = cx(1) + size(img_scale, 2);
    end
    segm31cell{i} = zeros(size(img, 1), size(img, 2));
    segm31cell{i}(cy(1) + 1 : cy(2), cx(1) + 1: cx(2)) = img_scale;
    segm31cell{i} = uint8(segm31cell{i});
end

% Save segm31 and cropsegm for all images in the set
for i = 1:N
    disp(i);
    [~, bn] = fileparts(annot{1}{i});
    nm = bn(1:5);
    infofile = fullfile(rt, ['data/' setname '_up/dummy'],   [nm '_info.mat']);
    segmfile = fullfile(rt, ['data/' setname '_up/dummy'],   [nm '_segm.mat']);
    cropsegm = cropsegmcell{i};
    segm31 = segm31cell{i};
    if(exist(infofile, 'file'))
        save(infofile, 'cropsegm', '-append');
        if(exist(segmfile, 'file'))
            save(segmfile, 'segm31', '-append');
        else
            disp(i);
            save(segmfile, 'segm31');
        end
    end
end

%% Copy the pre-processed data from the official split to the bmvc subset
all_sets = {train_bmvc, test_bmvc};
system(['mkdir -p ' rt 'data/train_bmvc/dummy/']);
system(['mkdir -p ' rt 'data/test_bmvc/dummy/']);
for j = 1:length(all_sets)
    setname = all_sets{j};
    for i = 1:length(setname)
        nm = setname{i};
        if(exist([rt 'data/train_up/dummy/' nm '_shape.mat']))
            system(['cp ' rt 'data/train_up/dummy/' nm '_shape.mat ' rt 'data/train_bmvc/dummy/' nm '_shape.mat']);
            system(['cp ' rt 'data/train_up/dummy/' nm '_info.mat ' rt 'data/train_bmvc/dummy/' nm '_info.mat']);
            system(['cp ' rt 'data/train_up/dummy/' nm '_image.png ' rt 'data/train_bmvc/dummy/' nm '_image.png']);
            system(['cp ' rt 'data/train_up/dummy/' nm '_segm.mat ' rt 'data/train_bmvc/dummy/' nm '_segm.mat']);
        elseif(exist([rt 'data/val_up/dummy/' nm '_shape.mat']))
            system(['cp ' rt 'data/val_up/dummy/' nm '_shape.mat ' rt 'data/train_bmvc/dummy/' nm '_shape.mat']);
            system(['cp ' rt 'data/val_up/dummy/' nm '_info.mat ' rt 'data/train_bmvc/dummy/' nm '_info.mat']);
            system(['cp ' rt 'data/val_up/dummy/' nm '_image.png ' rt 'data/train_bmvc/dummy/' nm '_image.png']);
            system(['cp ' rt 'data/val_up/dummy/' nm '_segm.mat ' rt 'data/train_bmvc/dummy/' nm '_segm.mat']);
        elseif(exist([rt 'data/test_up/dummy/' nm '_shape.mat']))
            system(['cp ' rt 'data/test_up/dummy/' nm '_shape.mat ' rt 'data/test_bmvc/dummy/' nm '_shape.mat']);
            system(['cp ' rt 'data/test_up/dummy/' nm '_info.mat ' rt 'data/test_bmvc/dummy/' nm '_info.mat']);
            system(['cp ' rt 'data/test_up/dummy/' nm '_image.png ' rt 'data/test_bmvc/dummy/' nm '_image.png']);
            system(['cp ' rt 'data/test_up/dummy/' nm '_segm.mat ' rt 'data/test_bmvc/dummy/' nm '_segm.mat']);
        else
            disp(nm);
        end
    end
end

%% Copy the pre-processed data from the official split to the lsp subset
all_sets = {'train_up', 'val_up', 'test_up'};
system(['mkdir -p ' rt 'data/train_lsp/dummy/']);
system(['mkdir -p ' rt 'data/test_lsp/dummy/']);
for j = 1:length(all_sets);
    setstring = all_sets{j};
    setname = eval(setstring);
    if strcmp(setstring, 'train_up') || strcmp(setstring, 'val_up')
        lspset = 'train_lsp';
    elseif strcmp(setstring, 'test_up')
        lspset = 'test_lsp';
    end
    for i = 1:length(setname) % (first 523 lsp)
        nm = setname{i};
        fid = fopen([rt 'up-3d/' nm '_dataset_info.txt']);
        dinfo = textscan(fid, '%s');
        fclose(fid);
        if(strcmp(dinfo{1}{1}, 'lsp'))
            system(['cp ' rt 'data/' setstring '/dummy/' nm '_shape.mat ' rt 'data/' lspset '/dummy/' nm '_shape.mat']);
            system(['cp ' rt 'data/' setstring '/dummy/' nm '_info.mat ' rt 'data/' lspset '/dummy/' nm '_info.mat']);
            system(['cp ' rt 'data/' setstring '/dummy/' nm '_image.png ' rt 'data/' lspset '/dummy/' nm '_image.png']);
            system(['cp ' rt 'data/' setstring '/dummy/' nm '_segm.mat ' rt 'data/' lspset '/dummy/' nm '_segm.mat']);
        end
    end
end