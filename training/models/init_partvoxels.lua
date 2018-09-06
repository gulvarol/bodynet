function listModules(model)
    for k = 1, #model.modules do;
        print(k .. ' ' .. torch.typename(model.modules[k]));
    end
end

require 'cudnn'
require 'nngraph'
require 'cunn'

torch.setdefaulttensortype('torch.FloatTensor')

model = torch.load('t7/model_voxels_cmu.t7')

nParts = 7
divideBy = 7 -- 1 or 7

-- Modules that need modification

-- model.modules[41]
-- cudnn.SpatialConvolution(256 -> 128, 1x1)
-- weight: 128, 256, 1, 1
model.modules[41].weight = model.modules[41].weight:repeatTensor(nParts, 1, 1, 1):clone():div(divideBy):clone()
model.modules[41].gradWeight = model.modules[41].gradWeight:repeatTensor(nParts, 1, 1, 1):clone():div(divideBy):clone()
model.modules[41].bias = model.modules[41].bias:repeatTensor(nParts):clone():div(divideBy):clone()
model.modules[41].gradBias = model.modules[41].gradBias:repeatTensor(nParts):clone():div(divideBy):clone()
model.modules[41].nInputPlane = 256
model.modules[41].nOutputPlane = 128*nParts


-- model.modules[42].modules[3]
-- cudnn.SpatialConvolution(128 -> 128, 3x3, 1,1, 1,1)
-- weight: 128, 128, 3, 3

model.modules[42].modules[3].weight = model.modules[42].modules[3].weight:repeatTensor(nParts, nParts, 1, 1):clone():div(divideBy):clone()
model.modules[42].modules[3].gradWeight = model.modules[42].modules[3].gradWeight:repeatTensor(nParts, nParts, 1, 1):clone():div(divideBy):clone()
model.modules[42].modules[3].bias = model.modules[42].modules[3].bias:repeatTensor(nParts):clone():div(divideBy):clone()
model.modules[42].modules[3].gradBias = model.modules[42].modules[3].gradBias:repeatTensor(nParts):clone():div(divideBy):clone()
model.modules[42].modules[3].nInputPlane = 128*nParts
model.modules[42].modules[3].nOutputPlane = 128*nParts


-- model.modules[76]
-- cudnn.SpatialConvolution(256 -> 128, 1x1)
-- weight: 128, 256, 1, 1
model.modules[76].weight = model.modules[76].weight:repeatTensor(nParts, 1, 1, 1):clone():div(divideBy):clone()
model.modules[76].gradWeight = model.modules[76].gradWeight:repeatTensor(nParts, 1, 1, 1):clone():div(divideBy):clone()
model.modules[76].bias = model.modules[76].bias:repeatTensor(nParts):clone():div(divideBy):clone()
model.modules[76].gradBias = model.modules[76].gradBias:repeatTensor(nParts):clone():div(divideBy):clone()
model.modules[76].nInputPlane = 256
model.modules[76].nOutputPlane = 128*nParts


-- model.modules[77].modules[3]
-- cudnn.SpatialConvolution(128 -> 128, 3x3, 1,1, 1,1)
-- weight: 128, 128, 3, 3

model.modules[77].modules[3].weight = model.modules[77].modules[3].weight:repeatTensor(nParts, nParts, 1, 1):clone():div(divideBy):clone()
model.modules[77].modules[3].gradWeight = model.modules[77].modules[3].gradWeight:repeatTensor(nParts, nParts, 1, 1):clone():div(divideBy):clone()
model.modules[77].modules[3].bias = model.modules[77].modules[3].bias:repeatTensor(nParts):clone():div(divideBy):clone()
model.modules[77].modules[3].gradBias = model.modules[77].modules[3].gradBias:repeatTensor(nParts):clone():div(divideBy):clone()
model.modules[77].modules[3].nInputPlane = 128*nParts
model.modules[77].modules[3].nOutputPlane = 128*nParts

--model.modules[45]
--cudnn.SpatialConvolution(128 -> 256, 1x1)
--weight: 256, 128, 1, 1

model.modules[45].weight = model.modules[45].weight:repeatTensor(1, nParts, 1, 1):clone():div(divideBy):clone()
model.modules[45].gradWeight = model.modules[45].gradWeight:repeatTensor(1, nParts, 1, 1):clone():div(divideBy):clone()
model.modules[45].nInputPlane = 128*nParts
model.modules[45].nOutputPlane = 256


model:replace(function(module)
   if torch.typename(module) == 'cudnn.Sigmoid' then
      return nn.Identity()
   else
      return module
   end
end)

--model.modules[43] = nn.Identity()
--model.modules[78] = nn.Identity()

inp = nn.Identity()()
voxels = (model)(inp)
out = {}
for st = 1, 2 do
    curr_voxels = nn.SelectTable(st)(voxels)
    --nn.Transpose({dim1, dim2})
    curr_voxels = nn.View(nParts, 128, 128, 128):cuda()(curr_voxels)
    --curr_voxels = nn.Transpose({2, 5}):cuda()(curr_voxels)
    table.insert(out, curr_voxels)
end

modelPartVoxels = nn.gModule({inp}, out)
torch.save('t7/init_partvoxels.t7', modelPartVoxels)

input = {torch.rand(2, 34, 256, 256):cuda(), torch.rand(2, 16*19, 64, 64):cuda()}
output = modelPartVoxels:forward(input)
print(output)