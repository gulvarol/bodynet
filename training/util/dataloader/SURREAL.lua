local M = {}

local function getMatFile(path, str)
   return paths.dirname(path) .. '/' .. paths.basename(path, 'mp4') .. str .. '.mat'
end

-- Number of frames
local function getDuration(path) -- can get it with OpenCV instead cap:get{propId=7} --CV_CAP_PROP_FRAME_COUNT
   local zrot, nFrames
   if pcall(function() zrot = matio.load( getMatFile(path, '_info'), 'zrot'); nFrames = zrot:nElement() end) then
      return nFrames
   else
      if(opt.verbose) then print('Zrot not loaded ' .. path) end
      return 0
   end
end

local function loadCameraLoc(path)
   local camLoc--, camDist
   if pcall(function() camLoc = matio.load( getMatFile(path, '_info'), 'camLoc') end) then
   else
      if(opt.verbose) then print('CamLoc not loaded ' .. path) end; return nil
   end
   if(camLoc == nil) then; return nil; end
   return camLoc--[1]  
end

-- RGB
local function loadRGB(path, t)
   local cap = cv.VideoCapture{filename=path}
   cap:set{propId=1, value=t-1} --CV_CAP_PROP_POS_FRAMES
   --local w = cap:get{propId=3} --CV_CAP_PROP_FRAME_WIDTH
   --local h = cap:get{propId=4} --CV_CAP_PROP_FRAME_HEIGHT

   --local rgb_t = torch.zeros(3, h, w)
   local rgb_t
   if pcall(function() _, rgb_t = cap:read{}; rgb_t = rgb_t:permute(3, 1, 2):float()/255; rgb_t = rgb_t:index(1, torch.LongTensor{3, 2, 1}) end) then
   else
      if (opt.verbose) then print('Img not opened ' .. path) end
      return nil -- not opened
   end
   return rgb_t
end

-- Joints2D
local function loadJoints2D(path, t)
   local joints2D, vars
   if pcall(function() vars = matio.load( getMatFile(path, '_info'), 'joints2D') end) then
      -- [24 x 2] -- it was 0-based
      if pcall(function() joints2D = vars[{{}, {}, { t }}]:squeeze():t():add(1); joints2D = joints2D:index(1, torch.LongTensor(opt.jointsIx)) end) then -- Get joint indices we are interested in
      else print(path .. ' has weirdness (joints2D)' .. t); return nil end
      local zeroJoint2D = joints2D[{{}, {1}}]:eq(1):cmul(joints2D[{{}, {2}}]:eq(239)) -- Check if joints are all zeros.
      if zeroJoint2D:sum()/zeroJoint2D:nElement() == 1 then
         if(opt.verbose) then print('Skipping ' .. path .. '... (joints2D are all [0, 0])') end
         return nil
      end
   else
      if(opt.verbose) then print('Joints2D not loaded ' .. path) end
   end
   return joints2D
end

-- Joints3D
local function loadJoints3D(path, t)
   local joints3D, vars
   if pcall(function() vars = matio.load( getMatFile(path, '_info'), 'joints3D') end) then
      if pcall(function() joints3D = vars[{{}, {}, { t }}]:squeeze():t(); joints3D = joints3D:index(1, torch.LongTensor(opt.jointsIx))  end) then       -- [24 x 3]
      else print(path .. ' has weirdness (joints3D)' .. t); return nil end
      local zeroJoint3D = joints3D[{{}, {1}}]:eq(0):cmul(joints3D[{{}, {2}}]:eq(0)):cmul(joints3D[{{}, {3}}]:eq(0)) -- Check if joints are all zeros.
      if zeroJoint3D:sum()/zeroJoint3D:nElement() == 1 then
         if(opt.verbose) then print('Skipping ' .. path .. '... (joints3D are all [0, 0])') end
         return nil
      end
   else
      if(opt.verbose) then print('Joints3D not loaded ' .. path) end
   end
   return joints3D
end

local function rotateBody(zrot, pelvisRotVec)
   -- zrot: euler
   local Rpelvis = Rotations.rotvec2rotmat(pelvisRotVec:view(3, 1))
   local RzBody  = torch.Tensor({ {math.cos(zrot), -math.sin(zrot), 0},
                            {math.sin(zrot),  math.cos(zrot), 0},
                            {0, 0, 1}})
   local globRotMat = RzBody*Rpelvis
   local out = Rotations.rotmat2rotvec(globRotMat)
   return Rotations.rotmat2rotvec(globRotMat)
end

-- SMPL pose parameters
local function loadPose(path, t)
   local pose, zrot, vars
   -- Different pose representations for body joint rotations with respect to their parents 
   local poseRotMat, poseRotVec
   if pcall(function() vars = matio.load( getMatFile(path, '_info'), {'pose', 'zrot'}) end) then
      if pcall(function() poseRotVec = vars.pose[{{}, { t }}]:squeeze()   end) then -- [72] 
         if pcall(function() zrot = vars.zrot[t][1]  end) then -- [1] 
            -- Rotate root (pelvis) by zrot
            poseRotVec[{{1, 3}}] = rotateBody(zrot, poseRotVec[{{1, 3}}])
            poseRotMat = torch.Tensor(9*24)
            for j = 1, 24 do
               local rotmat = Rotations.rotvec2rotmat(poseRotVec[{{3*j-2,3*j}}]:view(3, 1))
               if(opt.supervision == 'poseRotMat') then
                  poseRotMat[{{9*j-8, 9*j}}] = rotmat:view(9, 1)
               end
            end
            if(opt.supervision == 'poseRotMat') then
               pose = poseRotMat
            elseif(opt.supervision == 'pose' or opt.supervision == 'poseshape') then
               pose = poseRotVec
            end
         else
            if(opt.verbose) then print(path .. ' has weirdness (zrot)' .. t) end; return nil
         end
      else
         if(opt.verbose) then print(path .. ' has weirdness (pose)' .. t) end; return nil
      end
   else
      if(opt.verbose) then print('Pose not loaded ' .. path) end; return nil
   end
   return pose
end

-- SMPL beta parameters
local function loadShape(path, t)
   local shape, vars --zrot, 
   if pcall(function() vars = matio.load( getMatFile(path, '_info'), 'shape') end) then
      if pcall(function() shape = vars[{{}, { t }}]:squeeze()   end) then    -- [10] 
      else print(path .. ' has weirdness (shape)' .. t); return nil end
   else
      if(opt.verbose) then print('Shape not loaded ' .. path) end; return nil
   end
   return shape
end

-- Voxels
local function loadVoxels(path, t)
   local fillstr = 'voxelsfill'

   local voxels
   if pcall(function() voxels = matio.load( getMatFile(path, '_' .. fillstr), fillstr .. '_' .. t);
                       voxels = voxels:contiguous():permute(2, 1, 3)
                       voxels = image.flip(voxels:contiguous(), 2) -- ByteTensor()
                       end) then
   else
      if(opt.verbose) then print('Voxels not loaded ' .. path) end; return nil
   end
   return voxels
end

-- Part voxels
local function loadPartVoxels(path, t)
   local fillstr = 'partvoxelsfill'

   local voxels
   if pcall(function() voxels = matio.load( getMatFile(path, '_partvoxels'), fillstr .. '_' .. t);
                       voxels = voxels:byte() -- CharTensor() -> ByteTensor()
                       voxels = voxels:contiguous():permute(2, 1, 3)
                       voxels = image.flip(voxels:contiguous(), 2)
                       -- 0, 1, 2, 3, 4, 5, 6
                       end) then
   else
      if(opt.verbose) then print('Part voxels not loaded ' .. path) end; return nil
   end
   return voxels
end

-- Segmentation
local function loadSegm(path, t)
   local segm
   if pcall(function() segm = matio.load( getMatFile(path, '_segm'), 'segm_' .. t) end) then -- [240 x 320]
   else
      if(opt.verbose) then print('Segm not loaded ' .. path) end;  return nil 
   end
   if(segm == nil) then; return nil; end
   if(string.match(opt.supervision, 'segm')) then
      segm = changeSegmIx(segm, opt.segmIx) -- 0, 23/24? [240x320]
   else --if(string.match(opt.input, 'segm')) then -- including voxels supervision
      segm = changeSegmIx15(segm, opt.segmIx) -- [15x240x320]
   end

   return segm
end

-- Depth
local function loadDepth(path, t, dPelvis)
   local depth, out, pelvis, mask, nForeground, lowB, upB
   if pcall(function() depth = matio.load( getMatFile(path, '_depth'), 'depth_' .. t) end) then -- [240 x 320]
   else
      if(opt.verbose) then print('Depth not loaded ' .. path) end;  return nil, nil
   end
   if(depth == nil) then; return nil, nil; end
   out = torch.zeros(depth:size())
   mask = torch.le(depth, 1e+3)  -- background =1.0000e+10
   nForeground = mask:view(-1):sum()  -- #foreground pixels
   lowB = -(opt.depthClasses - 1)/2
   upB = (opt.depthClasses - 1)/2

   local fgix = torch.le(depth, 1e3)
   local bgix = torch.gt(depth, 1e3)
   if(opt.supervision == 'depth') then
      out[fgix] = torch.cmax(torch.cmin(torch.ceil(torch.mul(torch.add(depth[fgix], -dPelvis), 1/opt.stp)), upB), lowB) -- align and quantize
      out[bgix] = lowB-1 -- background class
      out = out:add(1+upB) -- so that it's between 0-19. It was [-10, -9, .. 0 .. 9].  
   end
   return out, nForeground 
end

M.getMatFile    = getMatFile
M.getDuration   = getDuration
M.loadCameraLoc = loadCameraLoc
M.loadRGB       = loadRGB
M.loadJoints2D  = loadJoints2D
M.loadJoints3D  = loadJoints3D
M.loadPose      = loadPose
M.loadShape     = loadShape
M.loadVoxels    = loadVoxels
M.loadPartVoxels= loadPartVoxels
M.loadSegm      = loadSegm
M.loadDepth     = loadDepth

return M