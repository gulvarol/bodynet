dofile('../training/util/eval.lua') -- getPreds2DCropped, getPreds3DCropped
dofile('../training/util/img.lua') -- getCenter, getScale, crop

sampleRes = 256

--Resize and crop
function cropCenter(rgb)
    local h = rgb:size(2)
    local w = rgb:size(3)

    local cropInfo = {}

    local inputSc, inputCropped
    if(h > w) then
        local newH = math.ceil(sampleRes*h/w)
        local st = math.ceil((newH - sampleRes)/2)
        inputSc = image.scale(rgb, sampleRes, newH)
        inputCropped = inputSc[{{}, {st, st+sampleRes-1}, {}}]:clone()
        cropInfo.sc = sampleRes/w
        cropInfo.x = st
    else
        local newW = math.ceil(sampleRes*w/h)
        local st = math.ceil((newW - sampleRes)/2)
        inputSc = image.scale(rgb, newW, sampleRes)
        inputCropped = inputSc[{{}, {}, {st+1, st+sampleRes}}]:clone()
        cropInfo.sc = sampleRes/h
        cropInfo.y = st
    end

    local p = 0
    local b = 0
    local inputPadded = torch.zeros(3, sampleRes+2*p, sampleRes+2*p)
    inputPadded[{{}, {b+1, b+sampleRes}, {p+1, p+sampleRes}}] = inputCropped:clone()

    local input = image.scale(inputPadded, sampleRes, sampleRes):view(1, 3, sampleRes, sampleRes)
    return input, cropInfo
end

function joints2Duncrop(joints2Dinitial, cropInfo)
    local out = joints2Dinitial:clone()
    if(cropInfo.x ~= nil) then
        out[{{}, {2}}] = out[{{}, {2}}] + cropInfo.x
    end
    
    if(cropInfo.y ~= nil) then
        out[{{}, {1}}] = out[{{}, {1}}] + cropInfo.y
    end
    out = out/cropInfo.sc
    
    return out
end

-- Crop
function cropTestImg(rgb, method, joints2D)
    local out, cropInfo
    if(method == 'center') then
        out, cropInfo = cropCenter(rgb)
    elseif(method == 'auto') then
        local scale = getScale(joints2D, dummy)
        local center = getCenter(joints2D)
        out = crop(rgb, center, scale, 0, sampleRes, 'bilinear')
    else
        print('What crop method?')
    end
    return out:view(1, 3, sampleRes, sampleRes), cropInfo
end

-- Apply 2D pose model
function applyHG2D(inputs, show)
    local joints2DInp = modelJoints2D:forward(inputs)
    local joints2DInp = joints2DInp[2] -- batchSize, 16, 64, 64
    local upsampling1 = nn.Sequential():add(nn.SpatialUpSamplingBilinear({oheight=sampleRes, owidth=sampleRes}))
    joints2DInp = upsampling1:cuda():forward(joints2DInp) -- batchSize, 16, 256, 256
    local predJoints2D = getPreds2DCropped(joints2DInp:float())

    return predJoints2D[1]
end

-- Apply models
function applyHG(inputs, show)
    local nStack = 2 -- 8

    local segmInp = modelSegm:forward(inputs)
    local joints2DInp = modelJoints2D:forward(inputs)
    local segmInp = segmInp[nStack] -- batchSize, 15, 64, 64
    local joints2DInp = joints2DInp[nStack] -- batchSize, 16, 64, 64

    local upsampling1 = nn.Sequential():add(nn.SpatialUpSamplingBilinear({oheight=sampleRes, owidth=sampleRes}))
    local upsampling2 = nn.Sequential():add(nn.SpatialUpSamplingBilinear({oheight=sampleRes, owidth=sampleRes}))
    joints2DInp = upsampling1:cuda():forward(joints2DInp) -- batchSize, 16, 256, 256
    segmInp = upsampling2:cuda():forward(segmInp) -- batchSize, 15, 256, 256
    segmInp = cudnn.SpatialSoftMax():cuda():forward(segmInp) -- normalize

    local rgbsegm15joints2DInp = nn.JoinTable(2):cuda():forward({inputs, segmInp, joints2DInp})
    local joints3DInp = modelJoints3D:forward(rgbsegm15joints2DInp)
    joints3DInp = joints3DInp[nStack]
    allInputs = {rgbsegm15joints2DInp, joints3DInp}
    local voxelsOut = modelVoxels:forward(allInputs)

    local dummy, predSegm = torch.max(segmInp[1]:float(), 1)
    local predJoints2D = getPreds2DCropped(joints2DInp:float())
    local predJoints3D = getPreds3DCropped(joints3DInp[1]:float(), 19)

    if(show) then
        wRGB = image.display({image=inputs, win=wRGB, legend='RGB'})
        wSegm = image.display({image=image.y2jet(predSegm:float()), win=wSegm, legend='SEGM'}) 
        wjoints2D = image.display({image=joints2DInp[1]:sum(1):squeeze(), win=wjoints2D, legend='2D POSE'})
    end

    return predSegm[1], predJoints2D[1], predJoints3D, voxelsOut[4][1]:float()
end

-- Apply end2end model
function applyEnd2End(inputs, show)
    local allOutputs = modelEnd2End:forward(inputs)

    local segmInp = allOutputs[6 + 1] -- batchSize, 15, 64, 64
    local joints2DInp = allOutputs[6 + 2] -- batchSize, 16, 64, 64
    local joints3DInp = allOutputs[6 + 3]
    local voxelsOut = allOutputs[6 + 4]:float()

    -- Upsample segm and joints2D
    local upsampling1 = nn.Sequential():add(nn.SpatialUpSamplingBilinear({oheight=sampleRes, owidth=sampleRes}))
    local upsampling2 = nn.Sequential():add(nn.SpatialUpSamplingBilinear({oheight=sampleRes, owidth=sampleRes}))
    joints2DInp = upsampling1:cuda():forward(joints2DInp) -- batchSize, 16, 256, 256
    segmInp = upsampling2:cuda():forward(segmInp) -- batchSize, 15, 256, 256
    segmInp = cudnn.SpatialSoftMax():cuda():forward(segmInp) -- normalize

    local dummy, predSegm = torch.max(segmInp[1]:float(), 1)
    local predJoints2D = getPreds2DCropped(joints2DInp:float())
    local predJoints3D = getPreds3DCropped(joints3DInp[1]:float(), 19)

    if(show) then
        wRGB = image.display({image=inputs, win=wRGB, legend='RGB'})
        wSegm = image.display({image=image.y2jet(predSegm:float()), win=wSegm, legend='SEGM'}) 
        wjoints2D = image.display({image=joints2DInp[1]:sum(1):squeeze(), win=wjoints2D, legend='2D POSE'})
    end

    return predSegm[1], predJoints2D[1], predJoints3D, voxelsOut[1]
end

-- Apply part voxels
function applyParts3D(inputs)
    local nStack = 2

    local segmInp = modelSegm:forward(inputs)
    local joints2DInp = modelJoints2D:forward(inputs)
    local segmInp = segmInp[nStack] -- batchSize, 15, 64, 64
    local joints2DInp = joints2DInp[nStack] -- batchSize, 16, 64, 64

    local upsampling1 = nn.Sequential():add(nn.SpatialUpSamplingBilinear({oheight=sampleRes, owidth=sampleRes}))
    local upsampling2 = nn.Sequential():add(nn.SpatialUpSamplingBilinear({oheight=sampleRes, owidth=sampleRes}))
    joints2DInp = upsampling1:cuda():forward(joints2DInp) -- batchSize, 16, 256, 256
    segmInp = upsampling2:cuda():forward(segmInp) -- batchSize, 15, 256, 256
    segmInp = cudnn.SpatialSoftMax():cuda():forward(segmInp) -- normalize

    local rgbsegm15joints2DInp = nn.JoinTable(2):cuda():forward({inputs, segmInp, joints2DInp})
    local joints3DInp = modelJoints3D:forward(rgbsegm15joints2DInp)
    joints3DInp = joints3DInp[nStack]
    allInputs = {rgbsegm15joints2DInp, joints3DInp}
    local partvoxelsOut = modelPartVoxels:forward(allInputs)

    return partvoxelsOut[4][1]:float()
end

-- Normalize input
function normalizeInput(inputs)
    mean = {0.510, 0.467, 0.411}
    std = {0.230, 0.235, 0.232}
    inputGPU = inputs:cuda()
    for j = 1, #mean do
        inputGPU[{{}, {j}, {}, {}}]:add(-mean[j])
        inputGPU[{{}, {j}, {}, {}}]:div(std[j])
    end
    return inputGPU
end
