if(opt.evaluate) then
    testLogger = optim.Logger(paths.concat(opt.save, 'evaluate.log'))
else
    testLogger = optim.Logger(paths.concat(opt.save, opt.testDir .. '.log'))
end

local timer = torch.Timer()
local batchNumber
local loss, pck, rmse, intersection, union, npos, pxlacc, iou, pe3d, pe3dMin, pe3dFair
local nanCntPck, nanCntPe3d
local intersectionvox, unionvox, iouvox
local intersectionproj, unionproj, iouproj
local intersectionprojsv, unionprojsv, iouprojsv
local intersectionpartvox, unionpartvox, ioupartvox

function test()
    local optimState 
    if(opt.evaluate) then
        print('==> Testing final predictions')
        if(opt.saveScores) then
            predFileTxt = io.open(paths.concat(opt.save, 'outputs.log'), "w")
        end
    else
    optimState = torch.load(paths.concat(opt.save, 'optimState_' .. epoch .. '.t7'))
  	print('==> validation epoch # ' .. epoch)
    end

    batchNumber = 0
    cutorch.synchronize()
    timer:reset()

    -- set to evaluate mode
    model:evaluate()

    loss         = 0
    pck          = 0
    rmse         = 0
    intersection = torch.Tensor(opt.segmClasses):zero()
    union        = torch.Tensor(opt.segmClasses):zero()
    npos         = torch.Tensor(opt.segmClasses):zero()
    pxlacc       = torch.Tensor(opt.segmClasses):zero()
    iou          = torch.Tensor(opt.segmClasses):zero()
    pe3d         = torch.Tensor(#opt.jointsIx):zero()
    pe3dMin      = 0
    pe3dFair     = 0
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

    for i= 1,(nTest/opt.batchSize) do -- nTest is set in 1_data.lua
        local indexStart = (i-1) * opt.batchSize + 1
        local indexEnd = math.min(nTest, indexStart + opt.batchSize - 1)
        donkeys:addjob(
            -- work to be done by donkey thread
            function()
                local inputs, labels, instances, indices = testLoader:get(indexStart, indexEnd)
                return inputs, labels, instances, indices
            end,
            -- callback that is run in the main thread once the work is done
            testBatch
        )
    end

    donkeys:synchronize()
    cutorch.synchronize()

    -- Performance measures:
    loss = loss / (nTest/opt.batchSize)
    pck = 100*pck / ((nTest/opt.batchSize) - nanCntPck)
    iou = 100*torch.cdiv(intersection, union)
    pxlacc = 100*torch.cdiv(intersection, npos)
    m_iou = iou[{{2, opt.segmClasses}}]:mean()
    m_pxlacc = pxlacc[{{2, opt.segmClasses}}]:mean()
    rmse = rmse / (nTest/opt.batchSize)
    pe3d = pe3d / ((nTest/opt.batchSize) - nanCntPe3d)
    pe3dFair = pe3dFair / ((nTest/opt.batchSize) - nanCntPe3d)
    pe3dMin = pe3dMin / ((nTest/opt.batchSize) - nanCntPe3d)
    m_pe3d = pe3d:mean()
    iouvox = 100* iouvox / (nTest/opt.batchSize) -- different from train.lua
    iouproj = 100* iouproj / (nTest/opt.batchSize)
    iouprojsv = 100*intersectionprojsv/unionprojsv
    ioupartvox = 100*torch.cdiv(intersectionpartvox, unionpartvox)
    m_ioupartvox = ioupartvox[{{2, opt.nParts3D}}]:mean()

    testLogger:add{
        ['epoch'] = epoch,
        ['loss'] = loss,
        ['pck'] = pck,
        ['pxlacc'] = table2str(pxlacc),
        ['iou'] = table2str(iou),
        ['m_iou'] = m_iou,
        ['m_pxlacc'] = m_pxlacc,
        ['rmse'] = rmse,
        ['pe3dvol'] = table2str(pe3d),
        ['pe3dFair'] = pe3dFair,
        ['pe3dMin'] = pe3dMin,
        ['m_pe3d'] = m_pe3d,
        ['iouvox'] = iouvox,
        ['iouprojfv'] = iouproj,
        ['iouprojsv'] = iouprojsv,
        ['ioupartvox'] = table2str(ioupartvox),
        ['m_ioupartvox'] = m_ioupartvox
    }
    if(not opt.evaluate) then
        opt.plotter:add('loss', 'test', epoch, loss)
        opt.plotter:add('pck', 'test', epoch, pck)
        opt.plotter:add('pxlacc', 'test', epoch, m_pxlacc)
        opt.plotter:add('iou', 'test', epoch, m_iou)
        opt.plotter:add('rmse', 'test', epoch, rmse)
        opt.plotter:add('pe3dvol', 'test', epoch, m_pe3d)
        opt.plotter:add('iouvox', 'test', epoch, iouvox)
        opt.plotter:add('iouprojfv', 'test', epoch, iouproj)
        opt.plotter:add('iouprojsv', 'test', epoch, iouprojsv)
        opt.plotter:add('ioupartvox', 'test', epoch, m_ioupartvox)
        print(string.format('Epoch: [%d] ', epoch))
    elseif(opt.saveScores) then predFileTxt:close() 
    end

    print(string.format('[TESTING SUMMARY] Total Time(s): %.2f \t'
        .. 'Loss: %.6f\t'
        .. 'PCK: %.2f\t'
        .. 'IOU: %.2f\t'
        .. 'PixAcc: %.2f\t'
        .. 'RMSE: %.2f\t'
        .. 'PE3Dvol: %.2f\t'
        .. 'PE3DFair: %.2f\t'
        .. 'PE3DMin: %.2f\t'
        .. 'IOUvox: %.2f\t'
        .. 'IOUprojFV: %.2f\t'
        .. 'IOUprojSV: %.2f\t'
        .. 'IOUpartvox: %.2f\t',
        timer:time().real, loss, pck, m_iou, m_pxlacc, rmse, m_pe3d, pe3dFair, pe3dMin, iouvox, iouproj, iouprojsv, m_ioupartvox))
    print('\n')

end -- of test()
-----------------------------------------------------------------------------

local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

function testBatch(inputsCPU, labelsCPU, instancesCPU, indicesCPU)
    batchNumber = batchNumber + opt.batchSize

    if pcall(function() n = #opt.inSize[1] end) then
        inputs = {}
        for cnt = 1, #opt.inSize do
            table.insert(inputs, torch.CudaTensor())
            -- transfer over to GPU
            inputs[cnt]:resize(inputsCPU[cnt]:size()):copy(inputsCPU[cnt])
        end
    else
        -- transfer over to GPU
        if(opt.applyHG == 'rgbsegm15joints2Djoints3D'
          or opt.applyHG == 'segm15joints3D') then
            local tmpInput = torch.CudaTensor()
            tmpInput:resize(inputsCPU:size()):copy(inputsCPU)
            inputs = applyHG(tmpInput, instancesCPU)
        else
            inputs:resize(inputsCPU:size()):copy(inputsCPU)
            inputs = applyHG(inputs)
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


    local outputs
  
    outputs = model:forward(inputs) -- table of nStack outputs

    local target
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

    -- Compute loss
    local err = criterion:forward(outputs, target)

    local pckBatch, pxlaccBatch, iouBatch, intersectionBatch, unionBatch, nposBatch, rmseBatch, pe3dBatch, pe3dFairBatch, pe3dMinBatch, iouvoxBatch, intersectionvoxBatch, unionvoxBatch, iouprojBatch, intersectionprojBatch, unionprojBatch, iouprojsvBatch, intersectionprojsvBatch, unionprojsvBatch, ioupartvoxBatch, intersectionpartvoxBatch, unionpartvoxBatch = evalPerf(inputsCPU, target, outputs, instancesCPU, indicesCPU)

    -- scoresCPU is defined after evalPerf, but labelsCPU already exists here

    str = string.format("PCK: %.2f,\tPixelAcc: %.2f,\tIOU: %.2f,\tRMSE: %.2f,\tPE3Dvol: %.2f,\tPE3DFair: %.2f,\tPE3DMin: %.2f,\tIOUvox: %.2f,\tIOUprojFV: %.2f,\tIOUprojSV: %.2f,\tIOUpartvox: %.2f", 100*pckBatch, 100*pxlaccBatch, 100*iouBatch, rmseBatch, pe3dBatch:mean(), pe3dFairBatch, pe3dMinBatch, 100*iouvoxBatch, 100*iouprojBatch, 100*iouprojsvBatch, 100*ioupartvoxBatch)

    if (pckBatch == pckBatch)                   then pck                 = pck + pckBatch else nanCntPck = nanCntPck + 1 end
    if (intersectionBatch == intersectionBatch) then intersection        = torch.add(intersection, intersectionBatch) end
    if (unionBatch == unionBatch)               then union               = torch.add(union, unionBatch) end
    if (nposBatch == nposBatch)                 then npos                = torch.add(npos, nposBatch)  end
    if (rmseBatch == rmseBatch)                 then rmse                = rmse + rmseBatch  end
    if (pe3dBatch:mean() == pe3dBatch:mean())   then pe3d                = torch.add(pe3d, pe3dBatch) else nanCntPe3d = nanCntPe3d + 1 end
    if (pe3dFairBatch == pe3dFairBatch)         then pe3dFair            = pe3dFair + pe3dFairBatch end
    if (pe3dMinBatch == pe3dMinBatch)           then pe3dMin             = pe3dMin + pe3dMinBatch end
    if (intersectionvoxBatch ~= 0)              then intersectionvox     = intersectionvox + intersectionvoxBatch end
    if (unionvoxBatch ~= 0)                     then unionvox            = unionvox + unionvoxBatch end
    if (intersectionprojBatch ~= 0)             then intersectionproj    = intersectionproj + intersectionprojBatch end
    if (unionprojBatch ~= 0)                    then unionproj           = unionproj + unionprojBatch end
    if (intersectionprojsvBatch ~= 0)           then intersectionprojsv  = intersectionprojsv + intersectionprojsvBatch end
    if (unionprojsvBatch ~= 0)                  then unionprojsv         = unionprojsv + unionprojsvBatch end
    if (intersectionpartvoxBatch ~= 0)          then intersectionpartvox = intersectionpartvox + intersectionpartvoxBatch end
    if (unionpartvoxBatch ~= 0)                 then unionpartvox        = unionpartvox + unionpartvoxBatch end
    iouvox = iouvox + iouvoxBatch
    iouproj = iouproj + iouprojBatch

    cutorch.synchronize()
    loss = loss + err

    if(opt.evaluate) then
        print(string.format('Testing [%d/%d] \t Loss %.8f \t %s', batchNumber, nTest, err, str))
    else
        print(string.format('Epoch: Testing [%d][%d/%d] \t Loss %.8f \t %s', epoch, batchNumber, nTest, err, str))
    end
end