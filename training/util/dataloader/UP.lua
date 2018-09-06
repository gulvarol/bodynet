local M = {}

local function parseRGB(path)
    return t
end

local function getMatFile(path, str)
   return paths.dirname(path) .. '/' .. string.sub(paths.basename(path), 1, 5) .. str .. '.mat'
end

local function loadFocalLength(path, t)
    local f
    if pcall(function() f = matio.load( getMatFile(path, '_info'), 'f') end) then
   else
      if(opt.verbose) then print('f not loaded ' .. path) end
   end
   return f[1]:float()
end

local function getCropInfo(path, t)
  local cropinfo, h, w, y1, y2, x1, x2
   if pcall(function()
        cropinfo = matio.load( getMatFile(path, '_info'), 'cropinfo')

        local crops = {}
        for i in string.gmatch(cropinfo:storage():string(), "%S+") do
            table.insert(crops, i)
        end

        h = tonumber(crops[1])
        w = tonumber(crops[2])
        y1 = tonumber(crops[3]) + 1
        y2 = tonumber(crops[4])
        x1 = tonumber(crops[5]) + 1
        x2 = tonumber(crops[6])

      end) then -- Get joint indices we are interested in
   else
      if(opt.verbose) then print('Cropinfo not loaded ' .. path) end
   end
   return {h, w, y1, y2, x1, x2} --h, w
end

local function loadCameraTrans(path, t)
    local t
    if pcall(function() t = matio.load( getMatFile(path, '_info'), 't') end) then
   else
      if(opt.verbose) then print('t not loaded ' .. path) end
   end
   local camLoc = t[1]:float()
         --camLoc = camLoc:index(1, torch.LongTensor({3, 2, 1}))
         --camLoc[2] =  -camLoc[2]
   return camLoc
end

-- RGB
local function loadRGB(path, t)
    local imgPath = path
    local rgb
    if pcall(function() rgb = image.load(imgPath); end) then
    else
        if (opt.verbose) then print('Img not opened ' .. path) end
        return nil -- jpg is not opened
    end
    return rgb
end

-- Joints2D
local function loadJoints2D(path, t)
   local joints2D, vars
   if pcall(function()
        -- [25 x 2] -- probably 0--based
        vars = matio.load( getMatFile(path, '_info'), {'joints2D', 'cropinfo'})

        local crops = {}
        for i in string.gmatch(vars.cropinfo:storage():string(), "%S+") do --strsplit(vars.cropinfo, ' ');
            table.insert(crops, i)
        end

        local h = tonumber(crops[1])
        local w = tonumber(crops[2])
        local y1 = tonumber(crops[3])+1
        local y2 = tonumber(crops[4])
        local x1 = tonumber(crops[5])+1
        local x2 = tonumber(crops[6])

        local new_w = x2 - x1 + 1
        local new_h = y2 - y1 + 1

        joints2D = torch.zeros(vars.joints2D:size())
        joints2D[{{}, {1}}] = torch.round(x1 - 1 + vars.joints2D[{{}, {1}}]*(new_h/h))
        joints2D[{{}, {2}}] = torch.round(y1 - 1 + vars.joints2D[{{}, {2}}]*(new_w/w))

        joints2D = joints2D:index(1, torch.LongTensor({1, 3, 2, 4, 6, 5, 7, 9, 8, 10, 12, 11, 13, 15, 14, 16, 18, 17, 20, 19, 22, 21, 24, 23})) -- Left/Right swapped 
        joints2D = joints2D:index(1, torch.LongTensor(opt.jointsIx)) 
      end) then -- Get joint indices we are interested in
   else
      if(opt.verbose) then print('Joints2D not loaded ' .. path) end
   end
   return joints2D
end

-- Joints3D
local function loadJoints3D(path, t)
	local joints3D
   if pcall(function() joints3D = matio.load( getMatFile(path, '_shape'), 'J_transformed') end) then
         joints3D = joints3D:index(1, torch.LongTensor({1, 3, 2, 4, 6, 5, 7, 9, 8, 10, 12, 11, 13, 15, 14, 16, 18, 17, 20, 19, 22, 21, 24, 23})) -- Left/Right swapped 
         joints3D = joints3D:index(1, torch.LongTensor(opt.jointsIx)):float()

   else
      if(opt.verbose) then print('Joints3D not loaded ' .. path) end
   end
   return joints3D
end

-- SMPL pose parameters
local function loadPose(path, t)
    local pose
    -- Different pose representations for body joint rotations with respect to their parents 
    local poseRotMat, poseRotVec
    if pcall(function() poseRotVec = matio.load( getMatFile(path, '_info'), 'pose'):view(-1) end) then -- [72]

        if(opt.supervision == 'pose' or opt.supervision == 'poseshape') then
            pose = poseRotVec
        else
            poseRotMat = torch.Tensor(9*24)
            for j = 1, 24 do
               local rotmat = Rotations.rotvec2rotmat(poseRotVec[{{3*j-2,3*j}}]:view(3, 1))
               if(opt.supervision == 'poseRotMat') then
                  poseRotMat[{{9*j-8, 9*j}}] = rotmat:view(9, 1)
               end
            end
            if(opt.supervision == 'poseRotMat') then
               pose = poseRotMat
            end
        end
   else
      if(opt.verbose) then print('Pose not loaded ' .. path) end; return nil
   end
   return pose
end

-- SMPL shape parameters
local function loadShape(path, t)
   local shape, vars --zrot, 
   if pcall(function() shape = matio.load( getMatFile(path, '_info'), 'shape') end) then
   else
      if(opt.verbose) then print('Shape not loaded ' .. path) end; return nil
   end
   return shape
	
end

local function loadSilhouette(path, t)
  local segm
  if pcall(function() segm = matio.load( getMatFile(path, '_segm'), 'segm7') end) then 
    segm = segm:gt(0) -- segm[{{2, -1}, {}, {}}]:sum(1):squeeze():gt(0)
    
  else
    if(opt.verbose) then print('Silhouette not loaded ' .. path) end;  return nil
  end 
  return segm
end

local function loadSegm(path, t)
    local segm, segm31
    --if pcall(function() segm = matio.load( getMatFile(path, '_segm'), 'segm1') end) then 
    --if pcall(function() segm = matio.load( getMatFile(path, '_segm'), 'segm7') end) then 
    if pcall(function() segm31 = matio.load( getMatFile(path, '_segm'), 'segm31') end) then 
        --local segmFile = '/home/gvarol/datasets/UP/up-s31/' .. string.sub(paths.basename(path), 1, 5) .. '_ann.png'
        --local segm31 = image.load(segmFile, 1, 'byte') -- 1, 513, 459
    else
        if(opt.verbose) then print('Segm not loaded ' .. path) end;  return nil 
    end

    --local segmix = {5, 4, 4, 3, 3, 2, 2, 2, 9, 10, 10, 10, 11, 8, 7, 7, 6, 6, 2, 2, 2, 12, 13, 13, 13, 14, 1, 1, 1, 1, 2}
    -- swap left/right    
    local segmix = {8, 7, 7, 6, 6, 2, 2, 2, 12, 13, 13, 13, 14, 5, 4, 4, 3, 3, 2, 2, 2, 9, 10, 10, 10, 11, 1, 1, 1, 1, 2}

    if(string.match(opt.supervision, 'segm')) then
        segm = changeSegmIx(segm31, segmix)
    else --if(string.match(opt.input, 'segm')) then -- including voxels supervision
        --segm15 = changeSegmIx15(segm, {1, 2, 3, 4, 5, 6, 7})  no 7?
        --segm15 = changeSegmIx15(segm, {1}) 
        segm = changeSegmIx15(segm31, segmix) -- [15x240x320]
    end

    return segm
end

-- Voxels
local function loadVoxels(path, t)

   local voxels
   if pcall(function() voxels = matio.load( getMatFile(path, '_shape'), 'voxelsfill');
                       voxels = voxels:contiguous():permute(2, 1, 3)
                       end) then
   else
      if(opt.verbose) then print('Voxels not loaded ' .. path) end; return nil
   end
   return voxels
end

-- Empty partvoxels
local function loadPartVoxels(path, t)
    return torch.Tensor(128, 128, 128):fill(1)
end

M.loadRGB         = loadRGB
M.loadJoints2D    = loadJoints2D
M.loadJoints3D    = loadJoints3D
M.loadPose        = loadPose
M.loadShape       = loadShape
M.loadSilhouette  = loadSilhouette
M.loadSegm        = loadSegm
M.loadVoxels      = loadVoxels
M.loadPartVoxels  = loadPartVoxels
M.loadFocalLength = loadFocalLength
M.getCropInfo     = getCropInfo
M.loadCameraTrans = loadCameraTrans

return M