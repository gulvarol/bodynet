local M = { }

function M.parse(arg)
   local defaultDir = '.'

    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Training script')
    cmd:text()
    cmd:text('Options:')
    ------------ General options --------------------
    cmd:option('-dataRoot',        paths.home .. '/datasets/', 'Home of datasets')
    cmd:option('-logRoot',         paths.home .. '/cnn_saves/', 'Home of logs')
    cmd:option('-datasetname',     'cmu',         'Name of the dataset. Options: cmu, H36M, UP')
    cmd:option('-dirName',         './',          'Path to experiment')
    cmd:option('-data',            './',          'Path to train/test splits')
    cmd:option('-save',            './save',      'subdirectory in which to log experiment')
    cmd:option('-cache',           './cache',     'subdirectory in which to cache data info')
    cmd:option('-plotter',         'plot',        'Path to the training curve.')
    cmd:option('-trainDir',        'train',       'Directory name of the train data')
    cmd:option('-testDir',         'val',         'Directory name of the test data')
    cmd:option('-manualSeed',      1,             'Manually set RNG seed')
    cmd:option('-GPU',             1,             'Default preferred GPU')
    cmd:option('-nGPU',            1,             'Number of GPUs to use by default')
    cmd:option('-backend',         'cudnn',       'Backend')
    cmd:option('-verbose',         false,         'Verbose')
    cmd:option('-show',            false,         'Visualize input/output')
    cmd:option('-continue',        false,         'Continue stopped training')
    cmd:option('-evaluate',        false,         'Final predictions or validation epochs')
    cmd:option('-saveScores',      true,          'Score saving')
    ------------- Data options ------------------------
    cmd:option('-nDonkeys',        8,             'number of donkeys to initialize (data loading threads)') 
    cmd:option('-sampleRes',       256,           'height/width resolution of input')
    cmd:option('-heatmapSize',     64,            'Heatmap size')
    cmd:option('-nVoxels',         128,           'Size of the voxel grid, default 128 x 128 x 128')
    cmd:option('-inSize',          {3, 256, 256}, 'Size of the input')
    cmd:option('-outSize',         {64, 64},      'Ground truth dimensions (table)')
    cmd:option('-input',           'rgb',         'Options: rgb, ...')    
    cmd:option('-supervision',     'voxels',      'Options: depth, segm, joints2D, joints3D, voxels, partvoxels, segm15joints2Djoints3D, segm15joints2Djoints3Dvoxels, segm15joints2Djoints3Dpartvoxels, pose, poseRotMat, shape, poseshape')
    cmd:option('-applyHG',         '',            'Apply pre-trained model on input. Options: segm15, joints2D, joints3D, segm15joints2D, rgbsegm15joints2D, segm15joints3D, rgbsegm15joints2Djoints3D')
    cmd:option('-extension',       {'mp4'},       'Video file extensions')
    cmd:option('-hflip',           true,          'Horizontal flip control random or lr_ambiguity')    
    cmd:option('-scale',           .25,           'Degree of scale augmentation')
    cmd:option('-rotate',          30,            'Degree of rotation augmentation')
    cmd:option('-whiten',          true,          'Mean subtraction and std division on input')
    cmd:option('-mix',             false,         'Mixed batch for gt + predicted input')
    cmd:option('-proj',            '',            'Segmentation or silhouette loss on the projection from front view or both front view and side view. Options: silhFV, segmFV, silhFVSV, segmFVSV')
    cmd:option('-scratchproj',     false,         'Train shape model from scratch')
    cmd:option('-protocol',        '',            'H36M protocols')
    cmd:option('-stp',             0.045,         'Depth quantization step')
    cmd:option('-depthClasses',    19,            'Number of depth bins for quantizing depth map (odd number)')
    cmd:option('-nParts3D',        7,             'Number of 3D body parts')
    cmd:option('-segmClasses',     15,            'Number of 2D segmentation classes')
    cmd:option('-mean',            {0.510, 0.467, 0.411}, 'Mean pixel values of the data.')
    cmd:option('-std',             {0.230, 0.235, 0.232}, 'Std pixel values of the data.')
    cmd:option('-segmIx',          {2, 12, 9, 2, 13, 10, 2, 14, 11, 2, 14, 11, 2, 2, 2, 1, 6, 3, 7, 4, 8, 5, 8, 5}, 'Indices for merging segmentation parts')
    cmd:option('-jointsIx',        {8, 5, 2, 3, 6, 9, 1, 7, 13, 16, 21, 19, 17, 18, 20, 22}, 'Joints ix')
    ------------- Loading options --------------
    cmd:option('-rgb',             false,         'Load rgb')
    cmd:option('-depth',           false,         'Load depth')
    cmd:option('-segm',            false,         'Load segmentation')
    cmd:option('-joints3D',        false,         'Load joints3D')
    cmd:option('-voxels',          false,         'Load voxels')
    cmd:option('-partvoxels',      false,         'Load partvoxels')
    ------------- Training options --------------------
    cmd:option('-nEpochs',         100,           'Number of total epochs to run')
    cmd:option('-epochSize',       2000,          'Number of batches per epoch')
    cmd:option('-epochNumber',     1,             'Manual epoch number (useful on restarts)')
    cmd:option('-batchSize',       4,             'mini-batch size (1 = pure stochastic)')
    cmd:option('-intsup',          true,          'Intermediate supervision between hourglasses')
    ---------- Optimization options ----------------------
    cmd:option('-LR',              1e-3,          'learning rate; if set, overrides default LR/WD recipe')
    cmd:option('-momentum',        0,             'momentum') -- 0.9
    cmd:option('-weightDecay',     0,             'weight decay') -- 5e-4
    cmd:option('-alpha',           0.99,          'Alpha for rmsprop')
    cmd:option('-epsilon',         1e-8,          'Epsilon for rmsprop')
    ---------- Model options ----------------------------------
    cmd:option('-netType',         'hg',          'Model type')
    cmd:option('-retrain',         'none',        'provide path to model to retrain with')
    cmd:option('-optimState',      'none',        'provide path to an optimState to reload from')
    cmd:option('-nStack',          2,             'Number of stacks in hg network')
    cmd:option('-nStackSegm',      2,             'Number of stacks in hg network')
    cmd:option('-nStackJoints2D',  2,             'Number of stacks in hg network')
    cmd:option('-nStackJoints3D',  2,             'Number of stacks in hg network')
    cmd:option('-nFeats',          256,           'Number of features in the hourglass')
    cmd:option('-nModules',        1,             'Number of residual modules at each location in the hourglass')
    cmd:option('-modelJoints2D',   'models/t7/model_joints2D.t7',      'Path to rgb->joints2D model')
    cmd:option('-modelSegm',       'models/t7/model_segm_cmu.t7',      'Path to rgb->segm model')
    cmd:option('-modelJoints3D',   'models/t7/model_joints3D_cmu.t7',  'Path to rgb+segm15+joints2D->joints3D model')
    cmd:option('-modelVoxels',     'models/t7/model_voxels_cmu.t7',    'Path to rgb+segm15+joints2D+joints3D->voxels model')
    cmd:text()

    local opt = cmd:parse(arg or {})
    -- add commandline specified options
    opt.save = paths.concat(opt.cache,
                            cmd:string('alexnet12', opt,
                                       {retrain=true, optimState=true, cache=true, data=true}))
    -- add date/time
    opt.save = paths.concat(opt.save, ',' .. os.date():gsub(' ',''))
    return opt
end

return M
