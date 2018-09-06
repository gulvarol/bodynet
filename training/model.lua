require 'nn'
require 'cunn'
require 'optim'
require 'cudnn'

-- Criterion

-- nStack * MSECriterion
if(opt.supervision == 'joints2D'
    or opt.supervision == 'joints3D') then -- nStacks * REGRESSION

    criterion = nn.ParallelCriterion()
    for st = 1, opt.nStack do -- 8
        criterion:add(nn.MSECriterion())
    end
-- MSECriterion
elseif(opt.supervision == 'pose'
        or opt.supervision == 'poseRotMat'
        or opt.supervision == 'shape'
        or opt.supervision == 'poseshape') then

    criterion = nn.MSECriterion()
-- nStack * SpatialCrossEntropyCriterion
elseif(opt.supervision == 'depth' or opt.supervision == 'segm') then -- nStacks * CLASSIFICATION
    criterion = nn.ParallelCriterion()
    for st = 1, opt.nStack do -- 8
        criterion:add(cudnn.SpatialCrossEntropyCriterion())
    end
-- nStack * BCECriterion
elseif(opt.supervision == 'voxels') then
    criterion = nn.ParallelCriterion()
    for st = 1, opt.nStack do
        if(opt.proj == 'silhFV') then
            local wts = torch.Tensor({10, 1})
            local wts_norm = wts / wts:sum()
            criterion:add(nn.BCECriterion(), wts_norm[1])
            criterion:add(nn.BCECriterion(), wts_norm[2])
        elseif(opt.proj == 'silhFVSV') then
            local wts = torch.Tensor({10, 1, 1})
            local wts_norm = wts / wts:sum()
            criterion:add(nn.BCECriterion(), wts_norm[1])
            criterion:add(nn.BCECriterion(), wts_norm[2])
            criterion:add(nn.BCECriterion(), wts_norm[3])
        else
            criterion:add(nn.BCECriterion())
        end
    end
-- nStack * VolumetricCrossEntropyCriterion
elseif(opt.supervision == 'partvoxels') then
    criterion = nn.ParallelCriterion()
    for st = 1, opt.nStack do 
        if(opt.proj == 'segmFV') then
            local wts
            --local wts = torch.Tensor({10000, 1})
            local wts = torch.Tensor({1, 1})
            local wts_norm = wts / wts:sum()
            criterion:add(cudnn.VolumetricCrossEntropyCriterion(), wts_norm[1])
            criterion:add(nn.BCECriterion(), wts_norm[2])
        elseif(opt.proj == 'segmFVSV') then
            --local wts = torch.Tensor({10000, 1, 1})
            local wts = torch.Tensor({1, 1, 1})
            local wts_norm = wts / wts:sum()
            criterion:add(cudnn.VolumetricCrossEntropyCriterion(), wts_norm[1])
            criterion:add(nn.BCECriterion(), wts_norm[2])
            criterion:add(nn.BCECriterion(), wts_norm[3])
        end
    end
-- END2END 3D POSE
elseif(opt.supervision == 'segm15joints2Djoints3D') then
    local wts = torch.Tensor({10000000, 1000, 1000000})
    local wts_norm = wts / wts:sum() 

    criterion = nn.ParallelCriterion()
    for st = 1, opt.nStack do
        criterion:add(cudnn.SpatialCrossEntropyCriterion(), wts_norm[1])
        criterion:add(nn.MSECriterion(), wts_norm[2])
        criterion:add(nn.MSECriterion(), wts_norm[3])
    end
-- END2END VOXELS
elseif(opt.supervision == 'segm15joints2Djoints3Dvoxels') then
    local wts = torch.Tensor({10000000, 1000, 1000000, 10})
    local wtsFV = torch.Tensor({10000000, 1000, 1000000, 10, 1})
    --local wtsFV = torch.Tensor({0, 0, 0, 10, 1}) -- experiment for no intermediate
    local wtsFVSV
    --if opt.datasetname == 'cmu' then
        wtsFVSV = torch.Tensor({10000000, 1000, 1000000, 10, 1, 1})
        --wtsFVSV = torch.Tensor({1, 1, 1, 1, 1, 1}) -- experiment for no balancing
    --elseif opt.datasetname == 'UP' then
    --    wtsFVSV = torch.Tensor({100, 1000, 100000, 100, 1, 1})
    --end
    --local wtsFVSV = torch.Tensor({0, 0, 0, 10, 1, 1}) -- experiment for no intermediate
    local wts_norm
    if(opt.proj == 'silhFV') then
        wts_norm = wtsFV / wtsFV:sum() 
    elseif(opt.proj == 'silhFVSV') then
        wts_norm = wtsFVSV / wtsFVSV:sum() 
    else
        wts_norm = wts / wts:sum()
    end

    print('Balancing with weights:')
    print(wts_norm)
    criterion = nn.ParallelCriterion()
    for st = 1, opt.nStack do
        criterion:add(cudnn.SpatialCrossEntropyCriterion(), wts_norm[1]) -- segmentation
        criterion:add(nn.MSECriterion(), wts_norm[2]) -- 2d pose
        criterion:add(nn.MSECriterion(), wts_norm[3]) -- 3d pose
        criterion:add(nn.BCECriterion(), wts_norm[4]) -- 3d voxels

        if(opt.proj == 'silhFV') then
            criterion:add(nn.BCECriterion(), wts_norm[5]) -- projection of voxels
        elseif(opt.proj == 'silhFVSV') then
            criterion:add(nn.BCECriterion(), wts_norm[5]) -- projection of voxels
            criterion:add(nn.BCECriterion(), wts_norm[6]) -- projection of voxels
        end
    end

-- END2END PART VOXELS
elseif(opt.supervision == 'segm15joints2Djoints3Dpartvoxels') then
    local wts = torch.Tensor({10000000, 1000, 1000000, 10})
    local wtsFV = torch.Tensor({10000000, 1000, 1000000, 10, 1})
    local wtsFVSV = torch.Tensor({10000000, 1000, 1000000, 10, 1, 1})

    local wts_norm
    if(opt.proj == 'segmFV') then
        wts_norm = wtsFV / wtsFV:sum() 
    elseif(opt.proj == 'segmFVSV') then
        wts_norm = wtsFVSV / wtsFVSV:sum() 
    else
        wts_norm = wts / wts:sum()
    end

    criterion = nn.ParallelCriterion()
    for st = 1, opt.nStack do
        criterion:add(cudnn.SpatialCrossEntropyCriterion(), wts_norm[1]) -- segmentation
        criterion:add(nn.MSECriterion(), wts_norm[2]) -- 2d pose
        criterion:add(nn.MSECriterion(), wts_norm[3]) -- 3d pose
        criterion:add(cudnn.VolumetricCrossEntropyCriterion(), wts_norm[4]) -- 3d voxels

        if(opt.proj == 'segmFV') then
            criterion:add(nn.BCECriterion(), wts_norm[5]) -- projection of voxels
        elseif(opt.proj == 'segmFVSV') then
            criterion:add(nn.BCECriterion(), wts_norm[5]) -- projection of voxels
            criterion:add(nn.BCECriterion(), wts_norm[6]) -- projection of voxels
        end
    end

else
    error('Criterion for this supervision type is not defined!')
end

-- Should come before createModel() because some of them use these.
opt.modelSegm = 'models/t7/model_segm_'.. opt.datasetname .. '.t7'
opt.modelJoints3D = 'models/t7/model_joints3D_'.. opt.datasetname .. '.t7'


-- Create Network
--    If preloading option is set, preload weights from existing models appropriately
--    If model has its own criterion, override.
if opt.retrain ~= 'none' then
    assert(paths.filep(opt.retrain), 'File not found: ' .. opt.retrain)
    print('Loading model from file: ' .. opt.retrain);
    model = loadDataParallel(opt.retrain, opt.nGPU)
else
    paths.dofile('models/' .. opt.netType .. '.lua')
    print('=> Creating model from file: models/' .. opt.netType .. '.lua')
    model = createModel(opt.nGPU) -- for the model creation code, check the models/ folder
    if opt.backend == 'cudnn' then
        require 'cudnn'
        cudnn.convert(model, cudnn)
    elseif opt.backend == 'cunn' then
        require 'cunn'
        model = model:cuda()
    elseif opt.backend ~= 'nn' then
        error'Unsupported backend'
    end
end
print('')

-- Whether to load pre-trained networks to provide inputs
if(opt.applyHG == 'segm15'
    or opt.applyHG == 'segm15voxelsavg') then
    print('Loading pre-trained segmentation network. ' .. opt.modelSegm)
    modelSegm = torch.load(opt.modelSegm)
    modelSegm:evaluate()
elseif(opt.applyHG == 'joints2D') then
    print('Loading pre-trained joints2D network. ' .. opt.modelJoints2D)
    modelJoints2D = torch.load(opt.modelJoints2D)
    modelJoints2D:evaluate()
elseif(opt.applyHG == 'segm15joints2D'
    or opt.applyHG == 'rgbsegm15joints2D'
    or opt.applyHG == 'rgbsegm15joints2Dvoxelsavg') then
    print('Loading pre-trained segmentation and joints2D networks. ' .. opt.modelSegm .. ' ' .. opt.modelJoints2D)
    modelSegm     = torch.load(opt.modelSegm)
    modelJoints2D = torch.load(opt.modelJoints2D)
    modelSegm:evaluate()
    modelJoints2D:evaluate()
elseif(opt.applyHG == 'rgbsegm15joints2Djoints3D'
    or opt.applyHG == 'segm15joints3D'
    or opt.applyHG == 'joints3D') then
    print('Loading pre-trained segmentation, joints2D and joints3D networks. ' .. opt.modelSegm .. ' ' .. opt.modelJoints2D .. ' ' .. opt.modelJoints3D)
    modelSegm     = torch.load(opt.modelSegm)
    modelJoints2D = torch.load(opt.modelJoints2D)
    modelJoints3D = torch.load(opt.modelJoints3D)
    modelSegm:evaluate()
    modelJoints2D:evaluate()
    modelJoints3D:evaluate()
end

print('=> Model')
--print(model)

print('=> Criterion')
--print(criterion)

-- Convert model to CUDA
print('==> Converting model and criterion to CUDA')
model:cuda()
criterion:cuda()

--cudnn.fastest = true
--cudnn.benchmark = true

collectgarbage()
