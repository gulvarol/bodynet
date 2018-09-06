require 'cudnn'
require 'cunn'
require 'env'
require 'image'
require 'nngraph'
local matio = require 'matio'
dofile('util.lua')

torch.setdefaulttensortype('torch.FloatTensor')

local dataset = 'UP' -- UP | cmu
-- sample_data/up/input.png | sample_data/surreal/input.png
local imgPath = 'sample_data/up/input.png'
local end2end = false
local autoCrop = true
local parts3D = true
local show = false

print('Loading pre-trained model(s)')
if end2end then
    modelEnd2End = torch.load('../training/models/t7/model_bodynet_' .. dataset .. '.t7')
    modelEnd2End:evaluate()
end
if not end2end or parts3D then
    modelSegm     = torch.load('../training/models/t7/model_segm_' .. dataset .. '.t7')
    modelJoints3D = torch.load('../training/models/t7/model_joints3D_' .. dataset .. '.t7')
    modelVoxels   = torch.load('../training/models/t7/model_voxels_FVSV_' .. dataset .. '.t7')
    modelSegm:evaluate()
    modelJoints3D:evaluate()
    modelVoxels:evaluate()
end
if autoCrop or not end2end then
    modelJoints2D = torch.load('../training/models/t7/model_joints2D.t7')
    modelJoints2D:evaluate()
end
if parts3D then
    modelPartVoxels = torch.load('../training/models/t7/model_partvoxels_FVSV_cmu.t7')
end

print('Loading input: ' .. imgPath)
local rgb = image.load(imgPath)
print('Center-cropping input')
local inputs, cropInfo = cropTestImg(rgb, 'center', {})
print('Normalizing input')
local inputGPU = normalizeInput(inputs)

if autoCrop then
    print('Predicting 2D pose')
    local joints2Dinitial = applyHG2D(inputGPU)
    local joints2Dinitial = joints2Duncrop(joints2Dinitial, cropInfo)
    -- Head joint is closer to the neck in smpl model --> avg over {neck, head}
    joints2Dinitial[10] = (joints2Dinitial[10] + joints2Dinitial[9]) / 2 
    print('Cropping input with predicted 2D pose')
    inputs, cropinfo = cropTestImg(rgb, 'auto', joints2Dinitial)
    print('Normalizing input')
    inputGPU = normalizeInput(inputs)
end

print('Forward pass')
local segmOut, joints2DOut, joints3DOut, voxelsOut
if end2end then
    segmOut, joints2DOut, joints3DOut, voxelsOut = applyEnd2End(inputGPU, show)
else
    segmOut, joints2DOut, joints3DOut, voxelsOut = applyHG(inputGPU, show)
end
if parts3D then
    partvoxelsOut = applyParts3D(torch.cat(inputGPU, inputGPU, 1)) --batch=2
end

print('Saving output: ' .. imgPath .. '.mat')
matio.save(imgPath .. '.mat',
    {rgb=inputs[1],
    segm=segmOut,
    joints2D=joints2DOut,
    joints3D=joints3DOut,
    voxels=voxelsOut,
    partvoxels=partvoxelsOut}
    )
print('Done')
