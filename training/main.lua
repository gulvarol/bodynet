require 'torch'
require 'cutorch'
require 'paths'
require 'xlua'
require 'optim'
require 'nn'
require 'nngraph'
require 'env'
paths.dofile('util/misc.lua')
paths.dofile('util/TrainPlotter.lua')
torch.setdefaulttensortype('torch.FloatTensor')

local opts = paths.dofile('opts.lua')
opt = opts.parse(arg)

--------------------------------
-- Determine paths, data options
--------------------------------
if(opt.datasetname == 'cmu') then
    opt.data = paths.concat(opt.dataRoot, 'SURREAL/data', opt.datasetname)
elseif(opt.datasetname == 'H36M') then
    opt.data = paths.concat(opt.dataRoot, 'H36M/data') 
    opt.protocol = 'p1'
    opt.testDir = 'test_p1'
    if(opt.evaluate) then
        opt.extension = {'jpg'}
    end
elseif(opt.datasetname == 'UP') then
    opt.data      = paths.concat(opt.dataRoot, 'UP/data')
    opt.extension = {'png'}
    --opt.testDir = 'test_lsp' -- test_up, test_lsp  test_bmvc
    opt.trainDir = 'train_up' --train_up train_lsp train_bmvc
end

local logDir = paths.concat(opt.logRoot, opt.datasetname)
opt.save = paths.concat(logDir, opt.dirName)
opt.cache = paths.concat(logDir, 'cache')
opt.plotter = TrainPlotter.new(paths.concat(opt.save, 'plot.json'))

os.execute('mkdir -p ' .. opt.save)
os.execute('mkdir -p ' .. opt.cache)

------------------------------------------------------
-- Determine input size, set which modalities to load
------------------------------------------------------
if(opt.input == 'rgb') then
    opt.inSize = {3, opt.sampleRes, opt.sampleRes}
    opt.rgb = true
elseif(opt.input == 'segm15') then
    opt.inSize = {15, opt.sampleRes, opt.sampleRes}
    opt.segm = true
    opt.whiten = false
elseif(opt.input == 'joints2D') then
    opt.inSize = {#opt.jointsIx, opt.sampleRes, opt.sampleRes}
    opt.whiten = false
elseif(opt.input == 'segm15joints2D') then
    opt.inSize = {(15 + #opt.jointsIx), opt.sampleRes, opt.sampleRes}
    opt.segm = true
    opt.whiten = false
elseif(opt.input == 'rgbsegm15joints2D') then
    opt.inSize = {(3 + 15 + #opt.jointsIx), opt.sampleRes, opt.sampleRes}
    opt.rgb = true
    opt.segm = true
    opt.whiten = false
elseif(opt.input == 'rgbsegm15joints2Djoints3D') then
    opt.inSize = {{(3 + 15 + #opt.jointsIx), opt.sampleRes, opt.sampleRes}, {opt.depthClasses*#opt.jointsIx, opt.heatmapSize, opt.heatmapSize}}
    opt.rgb = true
    opt.segm = true
    opt.joints3D = true
    opt.whiten = false
elseif(opt.input == 'segm15joints3D') then
    opt.inSize = {{15, opt.sampleRes, opt.sampleRes}, {opt.depthClasses*#opt.jointsIx, opt.heatmapSize, opt.heatmapSize}}
    opt.joints3D = true
    opt.whiten = false
elseif(opt.input == 'joints3D') then
    opt.inSize = {#opt.jointsIx*opt.depthClasses, opt.heatmapSize, opt.heatmapSize}
    opt.joints3D = true
    opt.sampleRes = opt.heatmapSize
    opt.whiten = false
end

------------------------------------------------------
-- Determine output size, set which modalities to load
------------------------------------------------------
if(opt.supervision == 'depth') then -- classification of depth bins
    opt.outSize = {opt.heatmapSize, opt.heatmapSize} -- 64 x 64
    opt.depth = true
    opt.joints3D = true
    opt.nOutChannels = opt.depthClasses + 1 -- 20
elseif(opt.supervision == 'segm') then
    opt.outSize = {opt.heatmapSize, opt.heatmapSize} -- 64 x 64
    opt.segm = true
    opt.nOutChannels = opt.segmClasses -- 15
elseif(opt.supervision == 'joints2D') then
    opt.outSize = {#opt.jointsIx, opt.heatmapSize, opt.heatmapSize} -- 16 x 64 x 64 
    opt.nOutChannels = #opt.jointsIx -- 16
elseif(opt.supervision == 'joints3D') then
    opt.outSize = {#opt.jointsIx * opt.depthClasses, opt.heatmapSize, opt.heatmapSize} -- (16 * 19) x 64 x 64
    opt.joints3D = true
    opt.nOutChannels = #opt.jointsIx * opt.depthClasses -- 16 * 19
elseif(opt.supervision == 'pose') then
    opt.outSize = {24*3, 1, 1}
    opt.pose = true
    opt.nOutChannels = 24*3
elseif(opt.supervision == 'poseRotMat') then
    opt.outSize = {24*9, 1, 1}
    opt.pose = true
    opt.nOutChannels = 24*9
elseif(opt.supervision == 'shape') then
    opt.outSize = {10, 1, 1}
    opt.shape = true
    opt.nOutChannels = 10
elseif(opt.supervision == 'poseshape') then
    opt.outSize = {24*3 + 10, 1, 1}
    opt.shape = true
    opt.pose = true
    opt.nOutChannels = 24*3 + 10
elseif(opt.supervision == 'voxels') then
    opt.outSize = {opt.nVoxels, opt.nVoxels, opt.nVoxels}
    opt.voxels = true
    opt.segm = true
    opt.nOutChannels = opt.nVoxels
elseif(opt.supervision == 'partvoxels') then
    opt.outSize = {opt.nVoxels, opt.nVoxels, opt.nVoxels}
    opt.partvoxels = true
    opt.segm = true
    opt.nOutChannels = opt.nParts3D * opt.nVoxels
elseif(opt.supervision == 'segm15joints2Djoints3D') then
    opt.outSize = {{opt.heatmapSize, opt.heatmapSize}, {#opt.jointsIx, opt.heatmapSize, opt.heatmapSize}, {#opt.jointsIx * opt.depthClasses, opt.heatmapSize, opt.heatmapSize}}
    opt.joints3D = true
    opt.segm = true
    -- nOutChannels
elseif(opt.supervision == 'segm15joints2Djoints3Dvoxels') then
    opt.outSize = {{opt.heatmapSize, opt.heatmapSize}, {#opt.jointsIx, opt.heatmapSize, opt.heatmapSize}, {#opt.jointsIx * opt.depthClasses, opt.heatmapSize, opt.heatmapSize}, {opt.nVoxels, opt.nVoxels, opt.nVoxels}}
    opt.joints3D = true
    opt.segm = true
    opt.voxels = true
elseif(opt.supervision == 'segm15joints2Djoints3Dpartvoxels') then
    opt.outSize = {{opt.heatmapSize, opt.heatmapSize}, {#opt.jointsIx, opt.heatmapSize, opt.heatmapSize}, {#opt.jointsIx * opt.depthClasses, opt.heatmapSize, opt.heatmapSize}, {opt.nVoxels, opt.nVoxels, opt.nVoxels}}
    opt.joints3D = true
    opt.segm = true
    opt.partvoxels = true
end

-------------------------
-- Determine architecture
-------------------------
-- (1) Standard hourglass
if(  opt.supervision == 'depth'
  or opt.supervision == 'segm'
  or opt.supervision == 'joints2D'
  or opt.supervision == 'joints3D') then
    opt.netType = 'hg' -- default
-- (2) Voxel prediction
elseif(opt.supervision == 'voxels' or opt.supervision == 'partvoxels') then
    -- Input: one-branch (64 x 64)
    if(opt.input == 'joints3D' or opt.applyHG == 'joints3D') then
        opt.netType = 'hg_voxels_b64'
    -- Input: two-branch (256 x 256) & (64 x 64)
    elseif(opt.input == 'segm15joints3D' or opt.applyHG == 'segm15joints3D'
        or opt.input == 'rgbsegm15joints2Djoints3D' or opt.applyHG == 'rgbsegm15joints2Djoints3D') then
        -- With projection
        if(opt.proj == 'silhFV' or opt.proj == 'silhFVSV' or opt.proj == 'segmFV' or opt.proj == 'segmFVSV') then
            opt.netType = 'hg_voxels_b256_b64_proj'
        -- Without projection
        else
            opt.netType = 'hg_voxels_b256_b64'
        end
    -- Input: one-branch (256 x 256)
    elseif(opt.input == 'segm15' or opt.applyHG == 'segm15'or opt.input == 'rgb') then
        opt.netType = 'hg_voxels_b256'
    end
-- (3) 3D pose prediction (end2end)
elseif(opt.supervision == 'segm15joints2Djoints3D') then
    opt.netType = 'hg_joints3D_end2end'
-- (4) Voxels/Partvoxels prediction (end2end)
elseif(opt.supervision == 'segm15joints2Djoints3Dvoxels' or opt.supervision == 'segm15joints2Djoints3Dpartvoxels') then
    opt.netType = 'hg_voxels_end2end'
-- (5) Vector prediction (fully connected at the end)
elseif(opt.supervision == 'pose'
  or opt.supervision == 'poseRotMat'
  or opt.supervision == 'shape'
  or opt.supervision == 'poseshape') then
    opt.nStack = 0 -- no stacked hourglass
    if(opt.input == 'segm15joints3D' or opt.applyHG == 'segm15joints3D'
        or opt.input == 'rgbsegm15joints2Djoints3D' or opt.applyHG == 'rgbsegm15joints2Djoints3D') then
        -- Two inputs concatenated 
        opt.netType = 'hg_fc_b256_b64'
    else
        opt.netType = 'hg_fc_b256'
    end
else
    error('Network architecture for this supervision type is not defined!')
end

---------------------------------------------------------------------------
-- Disable augmentations for which the GT augmentations are not implemented
---------------------------------------------------------------------------
if(    opt.supervision == 'joints3D'
    or opt.supervision == 'pose'
    or opt.supervision == 'poseRotMat'
    or opt.supervision == 'poseshape'
    or opt.supervision == 'shape'
    or opt.supervision == 'voxels'
    or opt.supervision == 'segm15joints2Djoints3D'
    or opt.supervision == 'segm15joints2Djoints3Dpartvoxels'
    or opt.supervision == 'segm15joints2Djoints3Dvoxels'
    or opt.supervision == 'partvoxels') then
    opt.rotate = 0
    opt.scale = 0
    opt.hflip = false
end

--------------------
-- Continue training
--------------------
if(opt.continue) then --epochNumber has to be set for this option
    print('Continuing from epoch ' .. opt.epochNumber)
    opt.retrain = opt.save .. '/model_' .. opt.epochNumber -1 .. '.t7'
    opt.optimState = opt.save .. '/optimState_'.. opt.epochNumber -1  ..'.t7'
    local backupDir = opt.save .. '/delete' .. os.time()
    os.execute('mkdir -p ' .. backupDir)
    os.execute('cp ' .. opt.save .. '/train.log ' .. backupDir)
    os.execute('cp ' .. opt.save .. '/test.log ' .. backupDir)
    os.execute('cp ' .. opt.save .. '/val.log ' .. backupDir)
    os.execute('cp ' .. opt.save .. '/plot.json ' .. backupDir)
end

------------------------
-- Setup and print stuff
------------------------
print(opt)
torch.save(paths.concat(opt.save, 'opt' .. os.time() .. '.t7'), opt)
--cutorch.setDevice(opt.GPU) -- by default, use GPU 1
torch.manualSeed(opt.manualSeed)
print('LR ' .. opt.LR)
print('Saving everything to: ' .. opt.save)

paths.dofile('model.lua')
paths.dofile('data.lua')
paths.dofile('train.lua')
paths.dofile('test.lua')
paths.dofile('util/eval.lua')

print('Used augmentations:')
print('Hflip ' .. tostring(opt.hflip))
print('Rotate ' .. opt.rotate)
print('Scale ' .. opt.scale)

----------------
-- Training loop
----------------
if(opt.evaluate) then
    test()
else
    epoch = opt.epochNumber
    for i=1,opt.nEpochs do
        train()
        test()
        epoch = epoch + 1
    end
end
