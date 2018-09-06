local ffi=require 'ffi'
------ Some FFI stuff used to pass storages between threads ------------------
ffi.cdef[[
void THFloatStorage_free(THFloatStorage *self);
void THLongStorage_free(THLongStorage *self);
]]

function makeDataParallel(model, nGPU)
    if nGPU > 1 then
        print('converting module to nn.DataParallelTable')
        assert(nGPU <= cutorch.getDeviceCount(), 'number of GPUs less than nGPU specified')
        local model_single = model
        model = nn.DataParallelTable(1)
        for i=1, nGPU do
            cutorch.setDevice(i)
            model:add(model_single:clone():cuda(), i)
        end
        cutorch.setDevice(opt.GPU)
    end
    return model
end

local function cleanDPT(module)
    -- This assumes this DPT was created by the function above: all the
    -- module.modules are clones of the same network on different GPUs
    -- hence we only need to keep one when saving the model to the disk.
    local newDPT = nn.DataParallelTable(1)
    cutorch.setDevice(opt.GPU)
    newDPT:add(module:get(1), opt.GPU)
    return newDPT
end

function saveDataParallel(filename, model)
    if torch.type(model) == 'nn.DataParallelTable' then
        torch.save(filename, cleanDPT(model))
    elseif torch.type(model) == 'nn.Sequential' then
        local temp_model = nn.Sequential()
        for i, module in ipairs(model.modules) do
            if torch.type(module) == 'nn.DataParallelTable' then
                temp_model:add(cleanDPT(module))
            else
                temp_model:add(module)
            end
        end
        torch.save(filename, temp_model)
    else
        torch.save(filename, model)
        print('The saved model is not a Sequential or DataParallelTable module.')
    end
end

function loadDataParallel(filename, nGPU)
    local model = torch.load(filename)
    if torch.type(model) == 'nn.DataParallelTable' then
        return makeDataParallel(model:get(1):float(), nGPU)
    elseif torch.type(model) == 'nn.Sequential' then
        for i,module in ipairs(model.modules) do
            if torch.type(module) == 'nn.DataParallelTable' then
                model.modules[i] = makeDataParallel(module:get(1):float(), nGPU)
            end
        end
        return model
    else
        print('The loaded model is not a Sequential or DataParallelTable module.')
        return model
    end
end

function setFloatStorage(tensor, storage_p)
    assert(storage_p and storage_p ~= 0, "FloatStorage is NULL pointer");
    local cstorage = ffi.cast('THFloatStorage*', torch.pointer(tensor:storage()))
    if cstorage ~= nil then
        ffi.C['THFloatStorage_free'](cstorage)
    end
    local storage = ffi.cast('THFloatStorage*', storage_p)
    tensor:cdata().storage = storage
end

function setLongStorage(tensor, storage_p)
    assert(storage_p and storage_p ~= 0, "LongStorage is NULL pointer");
    local cstorage = ffi.cast('THLongStorage*', torch.pointer(tensor:storage()))
    if cstorage ~= nil then
        ffi.C['THLongStorage_free'](cstorage)
    end
    local storage = ffi.cast('THLongStorage*', storage_p)
    tensor:cdata().storage = storage
end

function sendTensor(inputs)
    local size = inputs:size()
    local ttype = inputs:type()
    local i_stg =  tonumber(ffi.cast('intptr_t', torch.pointer(inputs:storage())))
    inputs:cdata().storage = nil
    return {i_stg, size, ttype}
end

function receiveTensor(obj, buffer)
    local pointer = obj[1]
    local size = obj[2]
    local ttype = obj[3]
    if buffer then
        buffer:resize(size)
        assert(buffer:type() == ttype, 'Buffer is wrong type')
    else
        buffer = torch[ttype].new():resize(size)      
    end
    if ttype == 'torch.FloatTensor' then
        setFloatStorage(buffer, pointer)
    elseif ttype == 'torch.LongTensor' then
        setLongStorage(buffer, pointer)
    else
        error('Unknown type')
    end
    return buffer
end

function listModules(model)
    --require 'cudnn'
    --require 'nngraph'
    --model = torch.load(modelfile)
    for k = 1, #model.modules do;
        print(k .. ' ' .. torch.typename(model.modules[k]));
    end
end

function getDir(dirName)
    dirs = paths.dir(dirName)
    table.sort(dirs, function (a,b) return a < b end)
    for i = #dirs, 1, -1 do
        if(dirs[i] == '.' or dirs[i] == '..') then
            table.remove(dirs, i)
        end
    end
    return dirs
end

function pause()
    io.stdin:read'*l'
end

function string.starts(String,Start)
   return string.sub(String,1,string.len(Start))==Start
end

function table2str ( v )
    if "string" == type( v ) then
        v = string.gsub( v, "\n", "\\n" )
        if string.match( string.gsub(v,"[^'\"]",""), '^"+$' ) then
            return "'" .. v .. "'"
        end
        return '"' .. string.gsub(v,'"', '\\"' ) .. '"'
    else
        return "table" == type( v ) and table.tostring( v ) or
        tostring( v )
    end
end

function table.key_to_str ( k )
    if "string" == type( k ) and string.match( k, "^[_%a][_%a%d]*$" ) then
        return k
    else
        return "[" .. table.val_to_str( k ) .. "]"
    end
end

function table.tostring( tbl )
    local result, done = {}, {}
    for k, v in ipairs( tbl ) do
        table.insert( result, table.val_to_str( v ) )
        done[ k ] = true
    end
    for k, v in pairs( tbl ) do
        if not done[ k ] then
            table.insert( result,
                table.key_to_str( k ) .. "=" .. table.val_to_str( v ) )
        end
    end
    return "{" .. table.concat( result, "," ) .. "}"
end


-- For predicted inputs to the network
-- instancesCPU is only used for rgbsegm15joints2Dvoxelspred input
function applyHG(inputs, instancesCPU)

    function predSegm(inputs)
        local segmInp = modelSegm:forward(inputs)
        segmInp = segmInp[opt.nStackSegm] -- batchSize, 15, 64, 64
        local upsampling = nn.Sequential():add(nn.SpatialUpSamplingBilinear({oheight=opt.sampleRes, owidth=opt.sampleRes}))
        segmInp = upsampling:cuda():forward(segmInp) -- batchSize, 15, 256, 256
        segmInp = cudnn.SpatialSoftMax():cuda():forward(segmInp) 

        if(opt.show) then
            print('Range segm pred ' .. segmInp:min() .. ' - ' .. segmInp:max())
            local dummy, pred = torch.max(segmInp[1], 1) -- 1 is batch
            local im = pred:float() -- 1 x 256 x 256
            im[1][1][1] = 1
            im[1][1][2] = opt.segmClasses
            wsegminp = image.display({image=image.y2jet(im), win=wsegminp, legend='PRED SEGM'})
        end
        return segmInp
    end

    function predJoints2D(inputs)
        local joints2DInp = modelJoints2D:forward(inputs)
        joints2DInp = joints2DInp[opt.nStackJoints2D] -- batchSize, 16, 64, 64
        local upsampling = nn.Sequential():add(nn.SpatialUpSamplingBilinear({oheight=opt.sampleRes, owidth=opt.sampleRes}))
        joints2DInp = upsampling:cuda():forward(joints2DInp) -- batchSize, 16, 256, 256

        if(opt.show) then
            print('Range joints2D pred ' .. joints2DInp:min() .. ' - ' .. joints2DInp:max())
            wjoints2Dinp = image.display({image=joints2DInp[1]:sum(1):squeeze(), win=wjoints2Dinp, legend='PRED 2D POSE'})
        end
        return joints2DInp
    end

    if(opt.applyHG == 'segm15') then
        inputs = predSegm(inputs)

    elseif(opt.applyHG == 'joints2D') then
        inputs = predJoints2D(inputs)

    elseif(opt.applyHG == 'segm15joints2D') then
        local segmInp = predSegm(inputs)
        local joints2DInp = predJoints2D(inputs)
        inputs = nn.JoinTable(2):cuda():forward({segmInp, joints2DInp})

    elseif(opt.applyHG == 'rgbsegm15joints2D') then
        local segmInp = predSegm(inputs)
        local joints2DInp = predJoints2D(inputs)
        inputs = nn.JoinTable(2):cuda():forward({inputs, segmInp, joints2DInp})

    elseif(opt.applyHG == 'rgbsegm15joints2Djoints3D') then
        local segmInp = predSegm(inputs)
        local joints2DInp = predJoints2D(inputs)
        local rgbsegm15joints2DInp = nn.JoinTable(2):cuda():forward({inputs, segmInp, joints2DInp})
        local joints3DInp = modelJoints3D:forward(rgbsegm15joints2DInp)
        joints3DInp = joints3DInp[opt.nStackJoints3D]
        inputs = {rgbsegm15joints2DInp, joints3DInp}

    elseif(opt.applyHG == 'joints3D') then
        local segmInp = predSegm(inputs)
        local joints2DInp = predJoints2D(inputs)
        local rgbsegm15joints2DInp = nn.JoinTable(2):cuda():forward({inputs, segmInp, joints2DInp})
        local joints3DInp = modelJoints3D:forward(rgbsegm15joints2DInp)
        joints3DInp = joints3DInp[opt.nStackJoints3D]
        inputs = joints3DInp:clone()

    elseif(opt.applyHG == 'segm15joints3D') then
        local segmInp = predSegm(inputs)
        local joints2DInp = predJoints2D(inputs)
        local rgbsegm15joints2DInp = nn.JoinTable(2):cuda():forward({inputs, segmInp, joints2DInp})
        local joints3DInp = modelJoints3D:forward(rgbsegm15joints2DInp)
        joints3DInp = joints3DInp[opt.nStackJoints3D]
        inputs = {segmInp, joints3DInp}
    end

    return inputs
end


