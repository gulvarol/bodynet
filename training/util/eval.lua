require 'image'
require 'gnuplot'
paths.dofile('camerautils.lua')
paths.dofile('drawutils.lua')
paths.dofile('img.lua')
local zm = 4 -- zoom

function calcDists(preds, label, normalize)
    local dists = torch.Tensor(preds:size(2), preds:size(1))
    local diff = torch.Tensor(2)
    for i = 1,preds:size(1) do
        for j = 1,preds:size(2) do
            if label[i][j][1] > 1 and label[i][j][2] > 1 then
                dists[j][i] = torch.dist(label[i][j],preds[i][j])/normalize[i]
            else
                dists[j][i] = -1
            end
        end
    end
    return dists
end

function tensorMax(input)
   local y = input:float():max()
   local i = input:float():eq(y):nonzero()
   return i[{{1}, {}}]
end

function getPreds3DCropped(hm, depthClasses)
   assert(hm:size():size() == 3, 'Input must be 3-D tensor')
   local nJoints = hm:size(1)/depthClasses
   local preds3D = torch.Tensor(nJoints, 3)
   hm = hm:view(nJoints, depthClasses, hm:size(2), hm:size(3)) -- 16 x 19 x 64 x 64
   for i = 1, nJoints do
      preds3D[{{i}, {}}] = tensorMax(hm[{{i}, {}, {}, {}}]:squeeze())
   end
   return preds3D
end

function getPreds2DCropped(hm)
    assert(hm:size():size() == 4, 'Input must be 4-D tensor')
    local max, idx = torch.max(hm:view(hm:size(1), hm:size(2), hm:size(3) * hm:size(4)), 3) -- idx !!
    local preds = torch.repeatTensor(idx, 1, 1, 2):float()
    --print(preds)
    preds[{{}, {}, 1}]:apply(function(x) return (x - 1) % hm:size(4) + 1 end)
    preds[{{}, {}, 2}]:add(-1):div(hm:size(3)):floor():add(1)

    local predMask = max:gt(0):repeatTensor(1, 1, 2):float()
    preds:add(-1):cmul(predMask):add(1)
    --print(preds)
    return preds
end

function getPreds2D(hms, center, scale)
    if hms:size():size() == 3 then hms = hms:view(1, hms:size(1), hms:size(2), hms:size(3)) end

    -- Get locations of maximum activations
    local max, idx = torch.max(hms:view(hms:size(1), hms:size(2), hms:size(3) * hms:size(4)), 3) -- idx !!
    local preds = torch.repeatTensor(idx, 1, 1, 2):float()
    preds[{{}, {}, 1}]:apply(function(x) return (x - 1) % hms:size(4) + 1 end)
    preds[{{}, {}, 2}]:add(-1):div(hms:size(3)):floor():add(.5)

--check
    local predMask = max:gt(0):repeatTensor(1, 1, 2):float()
    preds:add(-1):cmul(predMask):add(1)

    -- Get transformed coordinates
    local preds_tf = torch.zeros(preds:size())
    for i = 1,hms:size(1) do        -- Number of samples
        for j = 1,hms:size(2) do    -- Number of output heatmaps for one sample
            preds_tf[i][j] = transform(preds[i][j],center,scale,0,hms:size(3),true)
        end
    end

    return preds, preds_tf
end

function distAccuracy(dists, thr)
    -- Return percentage below threshold while ignoring values with a -1
    if not thr then thr = .5 end
    if torch.ne(dists,-1):sum() > 0 then
        return dists:le(thr):eq(dists:ne(-1)):sum() / dists:ne(-1):sum()
    else
        return -1
    end
end

function heatmapAccuracy(output, label, thr, idxs)
    -- Calculate accuracy according to PCK, but uses ground truth heatmap rather than x,y locations
    -- First value to be returned is average accuracy across 'idxs', followed by individual accuracies
    local preds = getPreds2DCropped(output)
    local gt = getPreds2DCropped(label)
    local dists = calcDists(preds, gt, torch.ones(preds:size(1))*opt.heatmapSize/10)
    local acc = {}
    local avgAcc = 0.0
    local badIdxCount = 0

    if not idxs then
        for i = 1,dists:size(1) do
            acc[i+1] = distAccuracy(dists[i])
    	    if acc[i+1] >= 0 then avgAcc = avgAcc + acc[i+1]
            else badIdxCount = badIdxCount + 1 end
        end
        acc[1] = avgAcc / (dists:size(1) - badIdxCount)
    else
        for i = 1,#idxs do
            acc[i+1] = distAccuracy(dists[idxs[i]])
	    if acc[i+1] >= 0 then avgAcc = avgAcc + acc[i+1]
            else badIdxCount = badIdxCount + 1 end
        end
        acc[1] = avgAcc / (#idxs - badIdxCount)
    end
    return unpack(acc)
end

function basicAccuracy(output, label, thr)
    -- Calculate basic accuracy
    if not thr then thr = .5 end -- Default threshold of .5
    output = output:view(output:numel())
    label = label:view(label:numel())

    local rounded_output = torch.ceil(output - thr):typeAs(label)
    local eql = torch.eq(label,rounded_output):typeAs(label)

    return eql:sum()/output:numel()
end


-- output: Bx5x64x64 probabilities for 5 classes, label: Bx64x64
function segmPerformance(output, labels)
    local nBatch = output:size(1)
    local nClasses = output:size(2) -- 15 for 2D, 7 for 3D segmentation
    local iou = torch.Tensor(nClasses):zero()
    local pixelacc = torch.Tensor(nClasses):zero()
    local intersection = torch.Tensor(nClasses):zero()
    local union = torch.Tensor(nClasses):zero()
    local npositives = torch.Tensor(nClasses):zero()
    for b = 1, nBatch do
        local label = labels[b] -- BUG DETECTED!!!!
        dummy, pred = torch.max(output[b], 1)
        for  cl = 1, nClasses do
            local ix = torch.eq(label, cl)
            local npos = ix:sum()
            local tp = torch.eq(label, pred:float()):cmul(ix):sum()
            local p = torch.eq(pred, cl):sum()

            intersection[cl] = intersection[cl] + tp
            union[cl] = union[cl] + (npos + p - tp )
            npositives[cl] = npositives[cl] + npos
            if(npos + p - tp ~= 0) then
                iou[cl] = iou[cl] + tp / (npos + p - tp )   -- precision.
            end
            if(npos ~= 0) then
                pixelacc[cl] = pixelacc[cl] + (tp / npos) -- not really accuracy? recall.
            end
        end
    end
    return iou[{{2, nClasses}}]:mean()/nBatch, iou, pixelacc[{{2, nClasses}}]:mean()/nBatch, pixelacc, intersection, union, npositives
end

-- output: Bx20x64x64, label:Bx64x64
function depthRMSE(output, label)
    local nBatch = output:size(1)
    local nClasses = output:size(2)
    local rmse = 0
    for b = 1, nBatch do
        local ix = torch.ne(label[b], 1):expandAs(label[b]) -- foreground pixels
        local nForeground = ix:sum()
        local dummy, pred = torch.max(output[b], 1)
        if(label[b][ix]:size():size() ~= 0) then -- not empty
            rmse = rmse + torch.sqrt(torch.mean(torch.pow(label[b][ix]:unfold(1, 2, 2) - pred[ix]:unfold(1, 2, 2):float(), 2)))
        else
            print('Does this happen?')
            -- counter of not evaluated images
        end
    end
    return rmse/nBatch
end

-- Unlike other evaluation functions, this one is not per batch. See the loop below in evalPerf func.
function poseError3D(predCoords3D_w, labelCoords3D_w, j3d)
    -- (Predicted, Ground truth)
    -- Pose error measured between our quantized bins versus predictions. It's fair because that's what it is trained on.
    local PE_fair = torch.Tensor(#opt.jointsIx)
    -- Pose error official. It is measured between the ground truth and our prediction.
    local PE_off = torch.Tensor(#opt.jointsIx)
    -- Pose error minimum. That is the lower bound we can get because of the quantization. Measured between ground truth and our labels used for training.
    local PE_min = torch.Tensor(#opt.jointsIx)
    for j = 1, #opt.jointsIx do
        PE_fair[j]    = 1000*math.sqrt(torch.pow(predCoords3D_w[j] - labelCoords3D_w[j], 2):sum()) -- 1000 * for converting from meters to milimeters
        PE_off[j]     = 1000*math.sqrt(torch.pow(predCoords3D_w[j] - j3d[j], 2):sum())
        PE_min[j]     = 1000*math.sqrt(torch.pow(labelCoords3D_w[j] - j3d[j], 2):sum())
    end
    return PE_fair, PE_off, PE_min
end

-- Used both for voxels and proj
-- output: Bx128x128x128, label: Bx128x128x128
function voxelsPerformance(output, labels)
    local nBatch = output:size(1)
    local iou = 0
    local intersection = 0
    local union = 0
    --local npositives = 0
    for b = 1, nBatch do
        local label = labels[b]
        local hardOut = output[b]:ge(0.5) -- (output[b]:max() - output[b]:min())/2
        local ix = torch.eq(label, 1)
        local npos = label:sum() --ix:sum()
        local tp = torch.eq(label, hardOut:float()):cmul(ix):sum()
        --local tp = torch.eq(label, hardOut:float()):cmul(label):sum()
        local p = torch.eq(hardOut, 1):sum()

        intersection = intersection + tp
        union = union + (npos + p - tp )
        --npositives = npositives + npos
        if(npos + p - tp ~= 0) then
            iou = iou + tp / (npos + p - tp )   -- precision.
            --print(string.format('%.2f, %d, %d, %.2f', iou, intersection, union, intersection/union))
        end
    end
    -- Equivalent
    --print(iou/nBatch)  
    --print(intersection/union)
    return iou/nBatch, intersection, union--, npositives
end

function evalPerf(inputsCPU, labelsTable, scoresTable, instancesCPU, indicesCPU)
    local iouBatch = 0
    local iouBatchParts = torch.Tensor(opt.segmClasses):zero()
    local pxlaccBatch = 0
    local pxlaccBatchParts = torch.Tensor(opt.segmClasses):zero()
    local intersectionBatch = torch.Tensor(opt.segmClasses):zero()
    local unionBatch = torch.Tensor(opt.segmClasses):zero()
    local npositives = torch.Tensor(opt.segmClasses):zero()
    local rmse = 0
    local pckBatch = 0
    local pe3dBatchParts = torch.Tensor(#opt.jointsIx):zero()
    local pe3dFairBatch = 0
    local pe3dMinBatch = 0
    local iouvoxBatch = 0
    local intersectionvoxBatch = 0
    local unionvoxBatch = 0
    local iouprojBatch = 0
    local intersectionprojBatch = 0
    local unionprojBatch = 0
    local iouprojsvBatch = 0
    local intersectionprojsvBatch = 0
    local unionprojsvBatch = 0
    local ioupartvoxBatch = 0
    local intersectionpartvoxBatch = torch.Tensor(opt.nParts3D):zero()
    local unionpartvoxBatch = torch.Tensor(opt.nParts3D):zero()

    if(opt.nStack > 0) then
        if(opt.proj == 'silhFV' or opt.proj == 'segmFV') then
            if(opt.supervision == 'segm15joints2Djoints3Dvoxels'
                or opt.supervision == 'segm15joints2Djoints3Dpartvoxels') then
                scoresCPU = scoresTable[5*(opt.nStack-1)+4]:float()
                labelsCPU = labelsTable[5*(opt.nStack-1)+4]:float()
            elseif(opt.supervision == 'voxels'
                or opt.supervision == 'partvoxels') then
                scoresCPU = scoresTable[2*(opt.nStack-1)+1]:float()
                labelsCPU = labelsTable[2*(opt.nStack-1)+1]:float()
            end
        elseif(opt.proj == 'silhFVSV' or opt.proj == 'segmFVSV') then
            if(opt.supervision == 'segm15joints2Djoints3Dvoxels'
                or opt.supervision == 'segm15joints2Djoints3Dpartvoxels') then
                scoresCPU = scoresTable[6*(opt.nStack-1)+4]:float()
                labelsCPU = labelsTable[6*(opt.nStack-1)+4]:float()
            elseif(opt.supervision == 'voxels'
                or opt.supervision == 'partvoxels') then
                scoresCPU = scoresTable[3*(opt.nStack-1)+1]:float()
                labelsCPU = labelsTable[3*(opt.nStack-1)+1]:float()
           end

        elseif(opt.supervision == 'segm15joints2Djoints3D') then
            scoresCPU = scoresTable[3*(opt.nStack-1)+3]:float()
            labelsCPU = labelsTable[3*(opt.nStack-1)+3]:float()
        else
            -- Take the last stack output
            scoresCPU = scoresTable[opt.nStack]:float()
            labelsCPU = labelsTable[opt.nStack]:float()
        end
    else
        -- No stack
        scoresCPU = scoresTable:float()
        labelsCPU = labelsTable:float()
    end

  local scoresJoints2D, labelsJoints2D, scoresSegm, labelsSegm, scoresDepth, labelsDepth
  local scoresJoints3D, labelsJoints3D
  if(opt.supervision == 'joints2D'
    or opt.supervision == 'segm15joints2Djoints3D'
    or opt.supervision == 'segm15joints2Djoints3Dvoxels'
    or opt.supervision == 'segm15joints2Djoints3Dpartvoxels') then

    if(opt.supervision == 'segm15joints2Djoints3D') then
        scoresJoints2D = scoresTable[3*(opt.nStack-1)+2]:float()
        labelsJoints2D = labelsTable[3*(opt.nStack-1)+2]:float()
    elseif(opt.supervision == 'segm15joints2Djoints3Dvoxels'
        or opt.supervision == 'segm15joints2Djoints3Dpartvoxels') then
        if(opt.proj == 'silhFV' or opt.proj == 'segmFV') then
            scoresJoints2D = scoresTable[5*(opt.nStack-1)+2]:float()
            labelsJoints2D = labelsTable[5*(opt.nStack-1)+2]:float()
        elseif(opt.proj == 'silhFVSV' or opt.proj == 'segmFVSV') then
            scoresJoints2D = scoresTable[6*(opt.nStack-1)+2]:float()
            labelsJoints2D = labelsTable[6*(opt.nStack-1)+2]:float()
        else
            scoresJoints2D = scoresTable[4*(opt.nStack-1)+2]:float()
            labelsJoints2D = labelsTable[4*(opt.nStack-1)+2]:float()
        end
    else
        scoresJoints2D = scoresCPU
        labelsJoints2D = labelsCPU
    end
    -- PCK 0.5, for joints 1, 16
    pckBatch = heatmapAccuracy(scoresJoints2D, labelsJoints2D, .5, {1,2,3,4,5,6,11,12,15,16})
    if(opt.show) then
        local ch = #opt.mean
        for cc=1, inputsCPU:size(2) do
            local mm = math.fmod(cc, ch)+1
            inputsCPU[{{}, {cc}, {}, {}}]=inputsCPU[{{}, {cc}, {}, {}}]:mul(opt.std[mm]):add(opt.mean[mm])
        end
      for i = 1, opt.batchSize do
        if(opt.input == 'rgb') then
            im = drawOutput(inputsCPU[{{i}, {1, 3}, {}, {}}]:squeeze(), scoresJoints2D[i], getPreds2DCropped(scoresJoints2D[{{i}, {}, {}, {}}]):squeeze()*4)
            imLabel = drawOutput(inputsCPU[{{i}, {1, 3}, {}, {}}]:squeeze(), labelsJoints2D[i], getPreds2DCropped(labelsJoints2D[{{i}, {}, {}, {}}]):squeeze()*4)
        end
        wOutputJoints2D = image.display({image=im, win=wOutputJoints2D, zoom=1, legend='PRED POSE 2D'})
        wLabelJoints2D = image.display({image=imLabel, win=wLabelJoints2D, zoom=1, legend='GT POSE 2D'})
      end
    end

  end
  
    if(opt.supervision == 'segm'
    or opt.supervision == 'segm15joints2Djoints3D'
    or opt.supervision == 'segm15joints2Djoints3Dvoxels'
    or opt.supervision == 'segm15joints2Djoints3Dpartvoxels') then
    
        if(opt.supervision == 'segm15joints2Djoints3D') then
            scoresSegm = scoresTable[3*(opt.nStack-1)+1]:float()
            labelsSegm = labelsTable[3*(opt.nStack-1)+1]:float()
        elseif(opt.supervision == 'segm15joints2Djoints3Dvoxels' or opt.supervision == 'segm15joints2Djoints3Dpartvoxels') then
            if(opt.proj == 'silhFV' or opt.proj == 'segmFV') then
                scoresSegm = scoresTable[5*(opt.nStack-1)+1]:float()
                labelsSegm = labelsTable[5*(opt.nStack-1)+1]:float()
            elseif(opt.proj == 'silhFVSV' or opt.proj == 'segmFVSV') then
                scoresSegm = scoresTable[6*(opt.nStack-1)+1]:float()
                labelsSegm = labelsTable[6*(opt.nStack-1)+1]:float()
            else
                scoresSegm = scoresTable[4*(opt.nStack-1)+1]:float()
                labelsSegm = labelsTable[4*(opt.nStack-1)+1]:float()
            end
        else
            scoresSegm = scoresCPU
            labelsSegm = labelsCPU
        end
        iouBatch, iouBatchParts, pxlaccBatch, pxlaccBatchParts, intersectionBatch, unionBatch, npositives = segmPerformance(scoresSegm, labelsSegm)
        if(opt.show) then
            require 'image'
            for i = 1, opt.batchSize do
                dummy, pred = torch.max(scoresSegm[i], 1) 
                im = pred:float()
                im[1][1][1] = 1
                im[1][1][2] = opt.segmClasses
                wOutputSegm = image.display({image=image.y2jet(im), win=wOutputSegm, zoom=zm, legend='PRED SEGM'})
            end
        end
    end

    if(opt.supervision == 'depth') then
        scoresDepth = scoresCPU
        labelsDepth = labelsCPU
        rmse = depthRMSE(scoresDepth, labelsDepth)
        if(opt.show) then
            require 'image'
            for i = 1, opt.batchSize do
                dummy, pred = torch.max(scoresDepth[i], 1) 
                im = pred:float() -- 1 x 64 x 64
                im[1][1][1] = 1
                im[1][1][2] = opt.depthClasses + 1
                wOutputDepth = image.display({image=image.y2jet(im), win=wOutputDepth, zoom=zm, legend='PRED DEPTH'})
            end
        end
    end


    if(opt.supervision == 'joints3D'
        or opt.supervision == 'segm15joints2Djoints3D'
        or opt.supervision == 'segm15joints2Djoints3Dvoxels'
        or opt.supervision == 'segm15joints2Djoints3Dpartvoxels') then

        if(opt.supervision == 'segm15joints2Djoints3D') then
            scoresJoints3D = scoresTable[3*(opt.nStack-1)+3]:float()
            labelsJoints3D = labelsTable[3*(opt.nStack-1)+3]:float()
        elseif(opt.supervision == 'segm15joints2Djoints3Dvoxels'
            or opt.supervision == 'segm15joints2Djoints3Dpartvoxels') then
            if(opt.proj == 'silhFV' or opt.proj == 'segmFV') then
                scoresJoints3D = scoresTable[5*(opt.nStack-1)+3]:float()
                labelsJoints3D = labelsTable[5*(opt.nStack-1)+3]:float()
            elseif(opt.proj == 'silhFVSV' or opt.proj == 'segmFVSV') then
                scoresJoints3D = scoresTable[6*(opt.nStack-1)+3]:float()
                labelsJoints3D = labelsTable[6*(opt.nStack-1)+3]:float()
            else
                scoresJoints3D = scoresTable[4*(opt.nStack-1)+3]:float()
                labelsJoints3D = labelsTable[4*(opt.nStack-1)+3]:float()
            end
        else
            scoresJoints3D = scoresCPU
            labelsJoints3D = labelsCPU
        end

        for i = 1, opt.batchSize do

            local predCoords3D = getPreds3DCropped(scoresJoints3D[i], opt.depthClasses) -- in 64x64x19 space
            local labelCoords3D = getPreds3DCropped(labelsJoints3D[i], opt.depthClasses)

            -- Redundant reloading of original 3D joint coordinates -- from real world in meters
            local j3d_c_a = instancesCPU[i].j3d_c_a
            local j2d = instancesCPU[i].joints2D 
            local RT = instancesCPU[i].RT 
            local R = instancesCPU[i].R 
            local T1 = instancesCPU[i].T1 
            local RT2 = instancesCPU[i].RT2 
            local R2 = instancesCPU[i].R2
            local T2 = instancesCPU[i].T2 
            local cropinfo = instancesCPU[i].cropinfo

            if(j3d_c_a == nil or j2d == nil or RT == nil or R == nil or T1 == nil or RT2 == nil or R2 == nil or T2 == nil
                or (opt.datasetname == 'UP' and cropinfo == nil) ) then
                j2d = torch.zeros(#opt.jointsIx, 2)
                if(opt.verbose) then print('nil at test') end  
            else

                if(opt.datasetname ~= 'H36M') then
                    j2d = j2d - 1 -- !! 
                end
                
                local K
                if(opt.datasetname == 'H36M') then
                    K = getIntrinsicH36M()
                elseif(opt.datasetname == 'cmu') then
                    K = getIntrinsicBlender()
                elseif(opt.datasetname == 'UP') then
                    K = instancesCPU[i].K
                else
                    print('No intrinsic defined')
                end

                local Kinv = torch.inverse(K)

                local j3d_q_64 = labelCoords3D:index(2, torch.LongTensor({3, 2, 1}))
                local pred_j3d_q_64 = predCoords3D:index(2, torch.LongTensor({3, 2, 1}))

                local scale = getScale(j2d+1, 240)
                local center = getCenter(j2d+1)

                -- Go back to the original image coordinates from the cropped coordinates
                paths.dofile('img.lua')
                local j2d_q = torch.Tensor(#opt.jointsIx, 2)
                local pred_j2d_q = torch.Tensor(#opt.jointsIx, 2)
                for j = 1,j3d_q_64:size(1) do
                    j2d_q[j] = transform(j3d_q_64[{{j}, {1, 2}}]:squeeze(),center,scale,0,64,true)
                    pred_j2d_q[j] = transform(pred_j3d_q_64[{{j}, {1, 2}}]:squeeze(),center,scale,0,64,true)

                    if(opt.datasetname == 'UP') then
                        local new_w = cropinfo[6] - cropinfo[5] + 1
                        local new_h = cropinfo[4] - cropinfo[3] + 1
                        j2d_q[j][1] = (j2d_q[j][1] - cropinfo[5] + 1) * cropinfo[1] / new_h
                        j2d_q[j][2] = (j2d_q[j][2] - cropinfo[3] + 1) * cropinfo[2] / new_w

                        pred_j2d_q[j][1] = (pred_j2d_q[j][1] - cropinfo[5] + 1) * cropinfo[1] / new_h
                        pred_j2d_q[j][2] = (pred_j2d_q[j][2] - cropinfo[3] + 1) * cropinfo[2] / new_w
                    end
                end

                -- Trying to recover j3d_c given 64x64 quantized 2d coordinates
                local middleZ = (opt.depthClasses + 1)/2 -- 10

                local rec3D_q = reconstruct3D(j2d_q-1, Kinv, bsxfunsum(-(j3d_q_64 - middleZ)*opt.stp, T1-T2))
                rec3D_q = bsxfunsum(rec3D_q, -rec3D_q[7]:clone())

                local pred_rec3D_q = reconstruct3D(pred_j2d_q-1, Kinv, bsxfunsum(-(pred_j3d_q_64 - middleZ)*opt.stp, T1-T2))
                pred_rec3D_q = bsxfunsum(pred_rec3D_q, -pred_rec3D_q[7]:clone())

                -- j3d and labelCoords3D_w should be close to each other. The official comparison is between j3d and predCoords3D_w.
                local PE_fair, PE_off, PE_min = poseError3D(pred_rec3D_q, rec3D_q, j3d_c_a)
                
                pe3dBatchParts = torch.add(pe3dBatchParts, PE_off)
                pe3dFairBatch = pe3dFairBatch + PE_fair:mean()
                pe3dMinBatch = pe3dMinBatch + PE_min:mean()

                if(opt.saveScores and opt.evaluate) then
                    predFileMat = paths.concat(opt.save, 'joints3D_' .. indicesCPU[i] .. '.mat')
                    local matio = require 'matio'
                    matio.save(predFileMat, {gt=j3d_c_a, pred=pred_rec3D_q})
                end

                if(opt.show) then

                    draw3DPose(rec3D_q, 4)
                    gnuplot.title('Orig GT 3D Pose')
                    draw3DPose(pred_rec3D_q, 3)
                    gnuplot.title('Orig PRED 3D Pose')

                    local im = draw2DPoseFrom3DJoints(rec3D_q:clone())  -- 3: x, 2: y
                    wLabel3Dxy = image.display({image=im, win=wLabel3Dxy, zoom=1, legend='GT 3D_xy Pose'})

                    local im = draw2DPoseFrom3DJoints(pred_rec3D_q:clone())
                    wOutput3Dxy = image.display({image=im, win=wOutput3Dxy, zoom=1, legend='PRED 3D_xy Pose'})
                    
                    wOutputJoints3Dhm = image.display({image=torch.cat({showJoints3DHeatmap(labelsJoints3D[i]), showJoints3DHeatmap(scoresJoints3D[i])}, 3), win=wOutputJoints3Dhm, zoom=2, legend='GT/PRED 3D POSE'})


                end -- if show
            end
        end -- for batch

        pe3dBatchParts = pe3dBatchParts/opt.batchSize
        pe3dFairBatch = pe3dFairBatch/opt.batchSize
        pe3dMinBatch = pe3dMinBatch/opt.batchSize     
    end -- if joints3D

-- PREDICTING FOREGROUND/BACKGROUND VOXELS
    if(opt.supervision == 'voxels'
    or opt.supervision == 'segm15joints2Djoints3Dvoxels') then

        if( opt.proj == 'silhFV') then
            if(opt.supervision == 'segm15joints2Djoints3Dvoxels') then
                -- There are 5 outputs per stack: segm, joints2D, joints3D, voxels, projFV
                scoresVoxels = scoresTable[5*(opt.nStack-1)+4]:float()
                labelsVoxels = labelsTable[5*(opt.nStack-1)+4]:float()

                scoresProjFV = scoresTable[5*(opt.nStack-1)+5]:float()
                labelsProjFV = labelsTable[5*(opt.nStack-1)+5]:float()

            elseif(opt.supervision == 'voxels') then
                -- There are 2 outputs per stack: voxels, projFV
                scoresVoxels = scoresTable[2*(opt.nStack-1)+1]:float() 
                labelsVoxels = labelsTable[2*(opt.nStack-1)+1]:float() 

                scoresProjFV = scoresTable[2*(opt.nStack-1)+2]:float()
                labelsProjFV = labelsTable[2*(opt.nStack-1)+2]:float()
            end

            if opt.show then
                for i = 1, opt.batchSize do
                    wOutputProjFV = image.display({image=scoresProjFV[i], win=wOutputProjFV, zoom=2, legend='PRED FV SILHOUETTE PROJ'})
                    wLabelProjFV  = image.display({image=labelsProjFV[i], win=wLabelProjFV , zoom=2, legend='GT FV SILHOUETTE PROJ'})
                end
            end
            iouprojBatch, intersectionprojBatch, unionprojBatch = voxelsPerformance(scoresProjFV, labelsProjFV)

        elseif( opt.proj == 'silhFVSV') then
            if(opt.supervision == 'segm15joints2Djoints3Dvoxels') then
                -- There are 6 outputs per stack: segm, joints2D, joints3D, voxels, projFV, projSV
                -- Voxel output (4th output of the last stack)
                scoresVoxels = scoresTable[6*(opt.nStack-1)+4]:float()
                labelsVoxels = labelsTable[6*(opt.nStack-1)+4]:float()
                -- Front view projection output (5th output of the last stack)
                scoresProjFV = scoresTable[6*(opt.nStack-1)+5]:float()
                labelsProjFV = labelsTable[6*(opt.nStack-1)+5]:float()
                -- Side view projection output (6th output of the last stack)
                scoresProjSV = scoresTable[6*(opt.nStack-1)+6]:float()
                labelsProjSV = labelsTable[6*(opt.nStack-1)+6]:float()

            elseif(opt.supervision == 'voxels') then
                -- There are 3 outputs per stack: voxels, projFV, projSV
                -- Voxel output (1st output of the last stack)
                scoresVoxels = scoresTable[3*(opt.nStack-1)+1]:float() 
                labelsVoxels = labelsTable[3*(opt.nStack-1)+1]:float() 
                -- Front view projection output (2nd output of the last stack)
                scoresProjFV = scoresTable[3*(opt.nStack-1)+2]:float()
                labelsProjFV = labelsTable[3*(opt.nStack-1)+2]:float()
                -- Side view projection output (3rd output of the last stack)
                scoresProjSV = scoresTable[3*(opt.nStack-1)+3]:float()
                labelsProjSV = labelsTable[3*(opt.nStack-1)+3]:float()
            end

            if opt.show then
                for i = 1, opt.batchSize do
                    wOutputProjFV = image.display({image=scoresProjFV[i], win=wOutputProjFV, zoom=2, legend='PRED FV SILHOUETTE PROJ'})
                    wLabelProjFV  = image.display({image=labelsProjFV[i], win=wLabelProjFV,  zoom=2, legend='GT FV SILHOUETTE PROJ'})
                    wOutputProjSV = image.display({image=scoresProjSV[i], win=wOutputProjSV, zoom=2, legend='PRED SV SILHOUETTE PROJ'})
                    wLabelProjSV  = image.display({image=labelsProjSV[i], win=wLabelProjSV,  zoom=2, legend='GT SV SILHOUETTE PROJ'})
                end
            end
            iouprojBatch,   intersectionprojBatch,   unionprojBatch   = voxelsPerformance(scoresProjFV, labelsProjFV)
            iouprojsvBatch, intersectionprojsvBatch, unionprojsvBatch = voxelsPerformance(scoresProjSV, labelsProjSV)

        else
            scoresVoxels = scoresCPU
            labelsVoxels = labelsCPU
        end

        if opt.show then
            for i = 1, opt.batchSize do
                wOutputVoxels = image.display({image=showVoxels(scoresVoxels[i]), win=wOutputVoxels, zoom=2, legend='PRED VOXELS'})
            end
        end -- for batch

        iouvoxBatch, intersectionvoxBatch, unionvoxBatch = voxelsPerformance(scoresVoxels, labelsVoxels)
    end

-- PREDICTING 3D BODY PART VOXELS (mostly copy-paste from above - voxels)
    if(opt.supervision == 'partvoxels'
    or opt.supervision == 'segm15joints2Djoints3Dpartvoxels') then

        if(opt.proj == 'segmFV') then
            if(opt.supervision == 'segm15joints2Djoints3Dpartvoxels') then
                -- There are 5 outputs per stack: segm, joints2D, joints3D, partvoxels, projFV
                scoresVoxels = scoresTable[5*(opt.nStack-1)+4]:float()
                labelsVoxels = labelsTable[5*(opt.nStack-1)+4]:float()

                scoresProjFV = scoresTable[5*(opt.nStack-1)+5]:float()
                labelsProjFV = labelsTable[5*(opt.nStack-1)+5]:float()

            elseif(opt.supervision == 'partvoxels') then
                -- There are 2 outputs per stack: voxels, projFV
                scoresVoxels = scoresTable[2*(opt.nStack-1)+1]:float() 
                labelsVoxels = labelsTable[2*(opt.nStack-1)+1]:float() 

                scoresProjFV = scoresTable[2*(opt.nStack-1)+2]:float()
                labelsProjFV = labelsTable[2*(opt.nStack-1)+2]:float()
            end

            if opt.show then
                for i = 1, opt.batchSize do
                    --print(scoresProjFV[i][1]:min() .. ' ' .. scoresProjFV[i][1]:max())
                    wOutputProjFV  = image.display({image=scoresProjFV[i][1], win=wOutputProjFV, zoom=2, legend='PRED FV SEGM PROJ (min)'})
                    wOutputProjFV4 = image.display({image=scoresProjFV[i][4], win=wOutputProjFV4, zoom=2, legend='PRED FV SEGM PROJ 4 (max)'})
                    --wOutputProjFV5 = image.display({image=scoresProjFV[i][5], win=wOutputProjFV5, zoom=2, legend='PRED FV SEGM PROJ 5 (max)'})
                    wOutputProjFV6 = image.display({image=scoresProjFV[i][6], win=wOutputProjFV6, zoom=2, legend='PRED FV SEGM PROJ 6 (max)'})
                    --wOutputProjFV7 = image.display({image=scoresProjFV[i][7], win=wOutputProjFV7, zoom=2, legend='PRED FV SEGM PROJ 7 (max)'})
                    wLabelProjFV   = image.display({image=labelsProjFV[i][1], win=wLabelProjFV , zoom=2, legend='GT FV SEGM PROJ (min)'}) 
                    wLabelProjFV4  = image.display({image=labelsProjFV[i][4], win=wLabelProjFV4 , zoom=2, legend='GT FV SEGM PROJ 4 (max)'}) 
                    --wLabelProjFV5  = image.display({image=labelsProjFV[i][5], win=wLabelProjFV5 , zoom=2, legend='GT FV SEGM PROJ 5 (max)'}) 
                    wLabelProjFV6  = image.display({image=labelsProjFV[i][6], win=wLabelProjFV6 , zoom=2, legend='GT FV SEGM PROJ 6 (max)'}) 
                    --wLabelProjFV7  = image.display({image=labelsProjFV[i][7]:float(), win=wLabelProjFV7 , zoom=2, legend='GT FV SEGM PROJ 7 (max)'})  
                end
            end

            iouprojBatch, intersectionprojBatch, unionprojBatch = voxelsPerformance(scoresProjFV, labelsProjFV)

        elseif(opt.proj == 'segmFVSV') then
            if(opt.supervision == 'segm15joints2Djoints3Dpartvoxels') then
                -- There are 6 outputs per stack: segm, joints2D, joints3D, partvoxels, projFV, projSV
                -- Voxel output (4th output of the last stack)
                scoresVoxels = scoresTable[6*(opt.nStack-1)+4]:float()
                labelsVoxels = labelsTable[6*(opt.nStack-1)+4]:float()
                -- Front view projection output (5th output of the last stack)
                scoresProjFV = scoresTable[6*(opt.nStack-1)+5]:float()
                labelsProjFV = labelsTable[6*(opt.nStack-1)+5]:float()
                -- Side view projection output (6th output of the last stack)
                scoresProjSV = scoresTable[6*(opt.nStack-1)+6]:float()
                labelsProjSV = labelsTable[6*(opt.nStack-1)+6]:float()

            elseif(opt.supervision == 'partvoxels') then
                -- There are 3 outputs per stack: voxels, projFV, projSV
                -- Voxel output (1st output of the last stack)
                scoresVoxels = scoresTable[3*(opt.nStack-1)+1]:float() 
                labelsVoxels = labelsTable[3*(opt.nStack-1)+1]:float() 
                -- Front view projection output (2nd output of the last stack)
                scoresProjFV = scoresTable[3*(opt.nStack-1)+2]:float()
                labelsProjFV = labelsTable[3*(opt.nStack-1)+2]:float()
                -- Side view projection output (3rd output of the last stack)
                scoresProjSV = scoresTable[3*(opt.nStack-1)+3]:float()
                labelsProjSV = labelsTable[3*(opt.nStack-1)+3]:float()
            end

            if opt.show then
                for i = 1, opt.batchSize do
                    wOutputProjFV  = image.display({image=scoresProjFV[i][1], win=wOutputProjFV, zoom=2, legend='PRED FV SEGM PROJ (min)'})
                    wOutputProjFV4 = image.display({image=scoresProjFV[i][4], win=wOutputProjFV4, zoom=2, legend='PRED FV SEGM PROJ 4 (max)'})

                    wLabelProjFV   = image.display({image=labelsProjFV[i][1], win=wLabelProjFV , zoom=2, legend='GT FV SEGM PROJ (min)'}) 
                    wLabelProjFV4  = image.display({image=labelsProjFV[i][4], win=wLabelProjFV4 , zoom=2, legend='GT FV SEGM PROJ 4 (max)'})  


                    wOutputProjSV  = image.display({image=scoresProjSV[i][1], win=wOutputProjSV, zoom=2, legend='PRED SV SEGM PROJ (min)'})
                    wOutputProjSV4 = image.display({image=scoresProjSV[i][4], win=wOutputProjSV4, zoom=2, legend='PRED SV SEGM PROJ 4 (max)'})

                    wLabelProjSV   = image.display({image=labelsProjSV[i][1], win=wLabelProjSV , zoom=2, legend='GT SV SEGM PROJ (min)'}) 
                    wLabelProjSV4  = image.display({image=labelsProjSV[i][4], win=wLabelProjSV4 , zoom=2, legend='GT SV SEGM PROJ 4 (max)'})   

                end
            end

            iouprojBatch,   intersectionprojBatch,   unionprojBatch   = voxelsPerformance(scoresProjFV, labelsProjFV)
            iouprojsvBatch, intersectionprojsvBatch, unionprojsvBatch = voxelsPerformance(scoresProjSV, labelsProjSV) 

        else
            scoresVoxels = scoresCPU
            labelsVoxels = labelsCPU
        end

        if opt.show then
            for i = 1, opt.batchSize do
                local dummy, pred = torch.max(scoresVoxels[i], 1)  -- [7, 128, 128, 128] -> [1, 128, 128, 128]
                scoresNorm = cudnn.VolumetricSoftMax():cuda():forward(scoresVoxels[i]:cuda()):float() 
                local im = pred:float():squeeze() -- [1, 128, 128, 128]-> [128, 128, 128]
                im[1][1][1] = 1
                im[1][1][2] = opt.nParts3D
                print('scoresVoxels')  
                print(scoresVoxels[i]:min() .. ' ' .. scoresVoxels[i]:max())
                print('scoresNorm')
                print(scoresNorm:min() .. ' ' .. scoresNorm:max()) 

                --scoresNorm = scoresVoxels[i]
                wOutputVoxels1 = image.display({image=showVoxels(scoresNorm[1], true), win=wOutputVoxels1, zoom=2, legend='PRED VOXELS 1 (BG)'})
                --wOutputVoxels2 = image.display({image=showVoxels(scoresNorm[2]), win=wOutputVoxels2, zoom=2, legend='PRED VOXELS 2 (HEAD)'})
                --wOutputVoxels3 = image.display({image=showVoxels(scoresNorm[3]), win=wOutputVoxels3, zoom=2, legend='PRED VOXELS 3 (TORSO)'})
                wOutputVoxels4 = image.display({image=showVoxels(scoresNorm[4]), win=wOutputVoxels4, zoom=2, legend='PRED VOXELS 4 (ARM)'})
                wOutputVoxels5 = image.display({image=showVoxels(scoresNorm[5]), win=wOutputVoxels5, zoom=2, legend='PRED VOXELS 5 (ARM)'})
                wOutputVoxels6 = image.display({image=showVoxels(scoresNorm[6]), win=wOutputVoxels6, zoom=2, legend='PRED VOXELS 6 (LEG)'})
                wOutputVoxels7 = image.display({image=showVoxels(scoresNorm[7]), win=wOutputVoxels7, zoom=2, legend='PRED VOXELS 7 (LEG)'})
                wOutputVoxels = image.display({image=showPartVoxels(im), win=wOutputVoxels, zoom=2, legend='PRED PARTVOXELS'})
            end
        end -- for batch

        ioupartvoxBatch, ioupartvoxBatchParts, pxlaccpartvoxBatch, pxlaccpartvoxBatchParts, intersectionpartvoxBatch, unionpartvoxBatch, npositivespartvox = segmPerformance(scoresVoxels, labelsVoxels)

    end

    if opt.show then
        sys.sleep(2)
    end

    
    if(opt.saveScores and opt.evaluate) then
        -- For batch
        for i=1,scoresCPU:size(1) do
            predFileMat = paths.concat(opt.save, 'outputs_' .. indicesCPU[i] .. '.mat')
            local matio = require 'matio'
            local gt, pred
            matio.save(predFileMat, {gt=labelsCPU[i]:squeeze(), pred=scoresCPU[i]:squeeze()})

        end
    end
  
    return pckBatch, pxlaccBatch, iouBatch, intersectionBatch, unionBatch, npositives, rmse, pe3dBatchParts, pe3dFairBatch, pe3dMinBatch, iouvoxBatch, intersectionvoxBatch, unionvoxBatch, iouprojBatch, intersectionprojBatch, unionprojBatch, iouprojsvBatch, intersectionprojsvBatch, unionprojsvBatch, ioupartvoxBatch, intersectionpartvoxBatch, unionpartvoxBatch
end