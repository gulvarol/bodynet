-- Setup a reused optimization state. If needed, reload from disk
if not optimState then
    optimState = {
        learningRate = opt.LR,
        learningRateDecay = 0.0,
        momentum = opt.momentum,
        weightDecay = opt.weightDecay,
        dampening = 0.0,
        alpha = opt.alpha,
        epsilon = opt.epsilon
    }
end

if opt.optimState ~= 'none' then
    assert(paths.filep(opt.optimState), 'File not found: ' .. opt.optimState)
    print('Loading optimState from file: ' .. opt.optimState)
    optimState = torch.load(opt.optimState)
    optimState.learningRate = opt.LR -- update LR
end

-- Logger
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
local batchNumber
local loss, pck, rmse, intersection, union, npos, pxlacc, iou, pe3d
local nanCntPck, nanCntPe3d
local intersectionvox, unionvox, iouvox
local intersectionproj, unionproj, iouproj
local intersectionprojsv, unionprojsv, iouprojsv
local intersectionpartvox, unionpartvox, ioupartvox

function train()
    print('==> doing epoch on training data:')
    print("==> online epoch # " .. epoch)
    trainEpochLogger = optim.Logger(paths.concat(opt.save, ("epoch_%d_train.log"):format(epoch))) 

    batchNumber = 0
    cutorch.synchronize()

    -- set the dropouts to training mode
    model:training()
    model:cuda()

    local tm = torch.Timer()
    loss         = 0
    pck          = 0
    rmse         = 0
    intersection = torch.Tensor(opt.segmClasses):zero()
    union        = torch.Tensor(opt.segmClasses):zero()
    npos         = torch.Tensor(opt.segmClasses):zero()
    pxlacc       = torch.Tensor(opt.segmClasses):zero()
    iou          = torch.Tensor(opt.segmClasses):zero()
    pe3d         = torch.Tensor(#opt.jointsIx):zero()
    intersectionvox = 0
    unionvox     = 0
    iouvox       = 0
    intersectionproj = 0
    unionproj    = 0
    iouproj      = 0
    intersectionprojsv = 0
    unionprojsv  = 0
    iouprojsv    = 0
    nanCntPck    = 0
    nanCntPe3d   = 0

    intersectionpartvox = torch.Tensor(opt.nParts3D):zero()
    unionpartvox        = torch.Tensor(opt.nParts3D):zero()
    ioupartvox          = torch.Tensor(opt.nParts3D):zero()

    for i=1,opt.epochSize do
        -- queue jobs to data-

        donkeys:addjob(
            -- the job callback (runs in data-worker thread)
            function()
                local inputs, labels, instances, indices = trainLoader:sample(opt.batchSize)
                return inputs, labels, instances, indices
            end,
            -- the end callback (runs in the main thread)
            trainBatch
        )
    end

    donkeys:synchronize()
    cutorch.synchronize()

    -- Performance measures:
    loss = loss / opt.epochSize
    pck = 100*pck / (opt.epochSize - nanCntPck)
    iou = 100*torch.cdiv(intersection, union)
    pxlacc = 100*torch.cdiv(intersection, npos)
    m_iou = iou[{{2, opt.segmClasses}}]:mean()
    m_pxlacc = pxlacc[{{2, opt.segmClasses}}]:mean()
    rmse = rmse / opt.epochSize
    pe3d = pe3d / (opt.epochSize - nanCntPe3d)
    m_pe3d = pe3d:mean()
    iouvox = 100*intersectionvox/unionvox -- different from test.lua
    iouproj =  100* iouproj / (opt.epochSize)
    iouprojsv = 100*intersectionprojsv/unionprojsv 
    ioupartvox = 100*torch.cdiv(intersectionpartvox, unionpartvox)
    m_ioupartvox = ioupartvox[{{2, opt.nParts3D}}]:mean()

    trainLogger:add{
        ['epoch'] = epoch,
        ['loss'] = loss,
        ['LR'] = optimState.learningRate,
        ['pck'] = pck,
        ['pxlacc'] = table2str(pxlacc),
        ['iou'] = table2str(iou),
        ['m_iou'] = m_iou,
        ['m_pxlacc'] = m_pxlacc,
        ['rmse'] = rmse,
        ['pe3dvol'] = table2str(pe3d),
        ['m_pe3dvol'] = m_pe3d,
        ['iouvox'] = iouvox,
        ['iouprojfv'] = iouproj,
        ['iouprojsv'] = iouprojsv,
        ['ioupartvox'] = table2str(ioupartvox),
        ['m_ioupartvox'] = m_ioupartvox
    }
    opt.plotter:add('LR', 'train', epoch, optimState.learningRate)
    opt.plotter:add('loss', 'train', epoch, loss)
    opt.plotter:add('pck', 'train', epoch, pck)
    opt.plotter:add('pxlacc', 'train', epoch, m_pxlacc)
    opt.plotter:add('iou', 'train', epoch, m_iou)
    opt.plotter:add('rmse', 'train', epoch, rmse)
    opt.plotter:add('pe3dvol', 'train', epoch, m_pe3d)
    opt.plotter:add('iouvox', 'train', epoch, iouvox)
    opt.plotter:add('iouprojfv', 'train', epoch, iouproj)
    opt.plotter:add('iouprojsv', 'train', epoch, iouprojsv)
    opt.plotter:add('ioupartvox', 'train', epoch, m_ioupartvox)

    print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f\t'
        .. 'Loss: %.6f \t'
        .. 'PCK: %.2f \t'
        .. 'IOU: %.2f \t'
        .. 'PixelAcc: %.2f \t'
        .. 'RMSE: %.2f \t'
        .. 'PE3Dvol: %.2f \t'
        .. 'IOUvox: %.2f \t'
        .. 'IOUprojFV: %.2f \t'
        .. 'IOUprojSV: %.2f \t'
        .. 'IOUpartvox: %.2f \t',
        epoch, tm:time().real, loss, pck, m_iou, m_pxlacc, rmse, m_pe3d, iouvox, iouproj, iouprojsv, m_ioupartvox))
    print('\n')
    collectgarbage()
    model:clearState()
    saveDataParallel(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), model)
    torch.save(paths.concat(opt.save, 'optimState_' .. epoch .. '.t7'), optimState)
end -- of train()
-------------------------------------------------------------------------------------------

-- GPU inputs (preallocate)
local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

local timer = torch.Timer()
local dataTimer = torch.Timer()

local parameters, gradParameters = model:getParameters()

-- Used by train() to train a single batch after the data is loaded.
function trainBatch(inputsCPU, labelsCPU, instancesCPU, indicesCPU)
    cutorch.synchronize()
    collectgarbage()
    local dataLoadingTime = dataTimer:time().real
    timer:reset()
    
    -- transfer over to GPU
    if(opt.mix) then
        local inputsCPU1 = inputsCPU[1] -- [q1, 3, 256, 256]
        local inputsCPU2 = inputsCPU[2] -- [q2, 15, 256, 256]
        local inputs1 = torch.CudaTensor()
        local inputs2
        inputs1:resize(inputsCPU1:size()):copy(inputsCPU1)
        if(opt.applyHG == 'rgbsegm15joints2Djoints3D'
        or opt.applyHG == 'segm15joints3D') then
            local inputs1Table = applyHG(inputs1, instancesCPU) -- [q1, 15, 256, 256]
            inputs2 = {}
            table.insert(inputs2, torch.CudaTensor())
            table.insert(inputs2, torch.CudaTensor())
            inputs2[1]:resize(inputsCPU2[1]:size()):copy(inputsCPU2[1])
            inputs2[2]:resize(inputsCPU2[2]:size()):copy(inputsCPU2[2])
            inputs = {torch.cat(inputs1Table[1], inputs2[1], 1), torch.cat(inputs1Table[2], inputs2[2], 1)}
        else
            inputs1 = applyHG(inputs1)
            inputs2 = torch.CudaTensor()
            inputs2:resize(inputsCPU2:size()):copy(inputsCPU2)
            inputs = torch.cat(inputs1, inputs2, 1) -- [q1+q2, 15, 256, 256]
        end

    else
        if pcall(function() n = #opt.inSize[1] end) then
            inputs = {}
            for cnt = 1, #opt.inSize do
                table.insert(inputs, torch.CudaTensor())
                inputs[cnt]:resize(inputsCPU[cnt]:size()):copy(inputsCPU[cnt])
            end
        else
            if(opt.applyHG == 'rgbsegm15joints2Djoints3D'
            or opt.applyHG == 'segm15joints3D') then
                local tmpInput = torch.CudaTensor()
                tmpInput:resize(inputsCPU:size()):copy(inputsCPU)
                inputs = applyHG(tmpInput, instancesCPU)
            else
                inputs:resize(inputsCPU:size()):copy(inputsCPU)
                inputs = applyHG(inputs, instancesCPU)
            end
        end
    end

    if pcall(function() n = #opt.outSize[1] end) then
        labels = {}
        for cnt = 1, #opt.outSize do
            table.insert(labels, torch.CudaTensor())
            labels[cnt]:resize(labelsCPU[cnt]:size()):copy(labelsCPU[cnt])
        end
    else
        labels:resize(labelsCPU:size()):copy(labelsCPU)
    end

    local err, outputs, target

    feval = function(x)
        model:zeroGradParameters()
        outputs = model:forward(inputs) -- table of nStack outputs

        if(opt.nStack > 1) then
            target = {}
            for st = 1, opt.nStack do
                table.insert(target, labels) -- same ground truth heatmap for all stacks

                local projFV, projSV
                
                if(opt.proj == 'silhFV' or opt.proj == 'silhFVSV') then
                    projFV = torch.CudaTensor()
                    projFV:resize(opt.batchSize, opt.nVoxels, opt.nVoxels)
                    for b = 1, opt.batchSize do
                        if(instancesCPU[b].silhFV) then
                            projFV[b]:copy(instancesCPU[b].silhFV)
                        else
                            print('Filling FV projection ground truth with zeros.')
                            projFV[b]:fill(0)
                        end
                    end
                    table.insert(target, projFV)
                end

                if(opt.proj == 'silhFVSV') then
                    projSV = torch.CudaTensor()
                    projSV:resize(opt.batchSize, opt.nVoxels, opt.nVoxels)
                    for b = 1, opt.batchSize do
                        if(instancesCPU[b].silhSV) then
                            projSV[b]:copy(instancesCPU[b].silhSV)
                        else
                            print('Filling SV projection ground truth with zeros.')
                            projSV[b]:fill(0)
                        end   
                    end
                    table.insert(target, projSV)
                end

                if(opt.proj == 'segmFV' or opt.proj == 'segmFVSV') then
                    projFV = torch.CudaTensor()
                    projFV:resize(opt.batchSize, opt.nParts3D, opt.nVoxels, opt.nVoxels)
                    for b = 1, opt.batchSize do
                        if(instancesCPU[b].segmFV) then
                            projFV[b]:copy(instancesCPU[b].segmFV)
                        else
                            print('Filling FV projection ground truth with zeros.')
                            projFV[b]:fill(0)
                        end
                    end
                    table.insert(target, projFV)
                end

                if(opt.proj == 'segmFVSV') then
                    projSV = torch.CudaTensor()
                    projSV:resize(opt.batchSize, opt.nParts3D, opt.nVoxels, opt.nVoxels)
                    for b = 1, opt.batchSize do
                        if(instancesCPU[b].segmSV) then
                            projSV[b]:copy(instancesCPU[b].segmSV)
                        else
                            print('Filling SV projection ground truth with zeros.')
                            projSV[b]:fill(0)
                        end
                    end
                    table.insert(target, projSV)
                end
            end
            if(opt.supervision == 'segm15joints2Djoints3D' or opt.supervision == 'segm15joints2Djoints3Dvoxels' or opt.supervision == 'segm15joints2Djoints3Dpartvoxels') then
                target = nn.FlattenTable():forward(target) -- for multi-task networks (i.e. when labels is a table)
            end
        else
            target = labels
        end

        err = criterion:forward(outputs, target)
        local gradOutputs = criterion:backward(outputs, target)

        -- This was used to watch the gradients with and without weighting.
        if false then
            local mattable = {}
            mattable.s1 = gradOutputs[1]:float()
            mattable.s2 = gradOutputs[2]:float()
            mattable.s3 = gradOutputs[3]:float()
            mattable.s4 = gradOutputs[4]:float()
            mattable.s5 = gradOutputs[5]:float()
            mattable.s6 = gradOutputs[6]:float()
            mattable.s7 = gradOutputs[7]:float()
            mattable.s8 = gradOutputs[8]:float()
            mattable.s9 = gradOutputs[9]:float()
            mattable.s10 = gradOutputs[10]:float()
            mattable.s11 = gradOutputs[11]:float()
            mattable.s12 = gradOutputs[12]:float()

            local matio = require 'matio'
            matio.save('not_weighted/data_' .. os.time() .. '.mat', mattable)
        end

        model:backward(inputs, gradOutputs)
        return err, gradParameters
    end

    optim.rmsprop(feval, parameters, optimState)

    local pckBatch, pxlaccBatch, iouBatch, intersectionBatch, unionBatch, nposBatch, rmseBatch, pe3dBatch, pe3dFairBatch, pe3dMinBatch, iouvoxBatch, intersectionvoxBatch, unionvoxBatch, iouprojBatch, intersectionprojBatch, unionprojBatch, iouprojsvBatch, intersectionprojsvBatch, unionprojsvBatch, ioupartvoxBatch, intersectionpartvoxBatch, unionpartvoxBatch = evalPerf(inputsCPU, target, outputs, instancesCPU, indicesCPU)

    str = string.format("PCK: %.2f,\tPixelAcc: %.2f,\tIOU: %.2f,\tRMSE: %.2f,\tPE3Dvol: %.2f,\tIOUvox: %.2f,\tIOUprojFV: %.2f,\tIOUprojSV: %.2f,\tIOUpartvox: %.2f", 100*pckBatch, 100*pxlaccBatch, 100*iouBatch, rmseBatch, pe3dBatch:mean(), 100*iouvoxBatch, 100*iouprojBatch, 100*iouprojsvBatch, 100*ioupartvoxBatch)

    if (pckBatch == pckBatch)                   then pck                   = pck + pckBatch else nanCntPck = nanCntPck + 1 end
    if (intersectionBatch == intersectionBatch) then intersection          = torch.add(intersection, intersectionBatch) end
    if (unionBatch == unionBatch)               then union                 = torch.add(union, unionBatch) end
    if (nposBatch == nposBatch)                 then npos                  = torch.add(npos, nposBatch)  end
    if (rmseBatch == rmseBatch)                 then rmse                  = rmse + rmseBatch  end
    if (pe3dBatch:mean() == pe3dBatch:mean())   then pe3d                  = torch.add(pe3d, pe3dBatch) else nanCntPe3d = nanCntPe3d + 1 end
    if (intersectionvoxBatch == intersectionvoxBatch) then intersectionvox = intersectionvox + intersectionvoxBatch end
    if (unionvoxBatch == unionvoxBatch)         then unionvox              = unionvox + unionvoxBatch end
    if (intersectionprojBatch ~= 0)             then intersectionproj      = intersectionproj + intersectionprojBatch end
    if (unionprojBatch ~= 0)                    then unionproj             = unionproj + unionprojBatch end
    if (intersectionprojsvBatch ~= 0)           then intersectionprojsv    = intersectionprojsv + intersectionprojsvBatch end
    if (unionprojsvBatch ~= 0)                  then unionprojsv           = unionprojsv + unionprojsvBatch end
    if (intersectionpartvoxBatch ~= 0)          then intersectionpartvox   = intersectionpartvox + intersectionpartvoxBatch end
    if (unionpartvoxBatch ~= 0)                 then unionpartvox          = unionpartvox + unionpartvoxBatch end
    iouproj = iouproj + iouprojBatch
  
    cutorch.synchronize()
    batchNumber = batchNumber + 1
    loss = loss + err
   
    print(('Epoch: [%d][%d/%d] Time: %.3f, Err: %.3f \t %s, \t LR: %.0e, \t DataLoadingTime %.3f'):format(
        epoch, batchNumber, opt.epochSize, timer:time().real, err, str,
        optimState.learningRate, dataLoadingTime))

    trainEpochLogger:add{
        ['BatchNumber'] = string.format("%d", batchNumber),
        ['Error'] = string.format("%.8f", err),
        ['LR'] = string.format("%.0e", optimState.learningRate)
    }
    dataTimer:reset()

end
