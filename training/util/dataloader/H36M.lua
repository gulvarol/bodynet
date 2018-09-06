local M = {}

local function parseRGBPath(path)
   local seqDir = paths.dirname(path) -- fullpath/rgb/jpg/Directions_1.xxxx.mp4_%05d.jpg

   -- Protocol 1 (S11 subject, every 64th frame, camera 2)
   if string.find(seqDir, "test_p1") then
      seqDir = string.gsub(seqDir, 'test_p1', 'test')
   -- Protocol 2 (S9, S11 subjects every 5th frame)
   elseif string.find(seqDir, "test_p2") then
      if(string.find(seqDir, "S11")) then
         seqDir = string.gsub(seqDir, 'test_p2', 'test')
      elseif(string.find(seqDir, "S9")) then
         seqDir = string.gsub(seqDir, 'test_p2', 'val')
      else
         print("Which test subject?")
      end
   -- Protocol 3 (S9, S11 subjects, camera 3, trial 1)
   elseif string.find(seqDir, "test_p3") then
      if(string.find(seqDir, "S11")) then
         seqDir = string.gsub(seqDir, 'test_p3', 'test')
      elseif(string.find(seqDir, "S9")) then
         seqDir = string.gsub(seqDir, 'test_p3', 'val')
      else
         print("Which test subject?")
      end
   else
     print("Which protocol??")
   end
   local seqNameAndClip = paths.basename(path, '.jpg') -- Directions_1.xxxx.mp4_%05d
   local seqName = string.sub(seqNameAndClip, 1, -11)
   local t = tonumber(string.sub(seqNameAndClip, -5, -1))
   local clipNo = math.ceil(t/100)
   local t1 = math.fmod(t + 99 , 100) +1
   local matPath = paths.concat(seqDir, '..', '..', '..', seqName .. string.format('_c%04d', clipNo) )
   return t1, matPath
end

local function getMatFile(path, str)
   local ext = paths.extname(path)
   if(ext == 'mp4') then
      return paths.dirname(path) .. '/' .. paths.basename(path, 'mp4') .. str .. '.mat'
   elseif(ext == 'jpg') then
      local t, matPath = parseRGBPath(path)
      return matPath .. str .. '.mat'
   end
end

-- Number of frames
local function getDuration(path) -- can get it with OpenCV instead
   local joints2D
   if pcall(function() joints2D = matio.load( getMatFile(path, '_info'), 'joints2Dh36m') end) then
      if(joints2D:nDimension() == 2) then
         return 1
      else
         return joints2D:size(3)
      end
   else
      if(opt.verbose) then print('Joints3D not loaded ' .. path) end
      return 0
   end
end

-- RGB -- t is redundant here
local function loadRGBjpg(path, t) 
   local imgPath = path
      local rgb_t
      if pcall(function() rgb_t = image.load(imgPath); end) then
      else
         if (opt.verbose) then print('Img not opened ' .. path) end
         return nil -- jpg is not opened
      end
   return rgb_t
end

-- RGB
local function loadRGB(path, t)
   -- if jpg use above
   if(paths.extname(path) == 'jpg') then
      return loadRGBjpg(path, t)
   else
      local cap = cv.VideoCapture{filename=path}
      cap:set{propId=1, value=t-1} --CV_CAP_PROP_POS_FRAMES
      --local w = cap:get{propId=3} --CV_CAP_PROP_FRAME_WIDTH
      --local h = cap:get{propId=4} --CV_CAP_PROP_FRAME_HEIGHT

      local rgb_t
      if pcall(function() _,rgb_t = cap:read{}; rgb_t = rgb_t:permute(3, 1, 2):float()/255; rgb_t = rgb_t:index(1, torch.LongTensor{3, 2, 1}) end) then
      else
         if (opt.verbose) then print('Img not opened ' .. path) end
         return nil -- not opened
      end
      return rgb_t
   end
end

-- Joints2D (H36M skeleton)
local function loadJoints2D(path, t) 
   local joints2D, vars
   if pcall(function() vars = matio.load( getMatFile(path, '_info'), 'joints2Dh36m') end) then
      if pcall(function() joints2D = vars[{{}, {}, { t }}]:squeeze():t(); end) then -- Get joint indices we are interested in
         joints2D = joints2D:index(1, torch.LongTensor({1, 3, 2, 4, 6, 5, 7, 9, 8, 10, 12, 11, 13, 15, 14, 16, 18, 17, 20, 19, 22, 21, 24, 23})) -- swap Left/Right
         joints2D = joints2D:index(1, torch.LongTensor(opt.jointsIx)) -- values between 1, 1000
      else print(path .. ' has weirdness (joints2D)' .. t); return nil end
   else
      if(opt.verbose) then print('Joints2D not loaded ' .. path) end
   end
   return joints2D
end

-- Joints3D (H36M skeleton)
local function loadJoints3D(path, t)
   local joints3D, vars
   if pcall(function() vars = matio.load( getMatFile(path, '_info'), 'joints3Dh36m') end) then
      if pcall(function() joints3D = vars[{{}, {}, { t }}]:squeeze():t(); end) then       -- [24 x 3]
         joints3D = joints3D:index(1, torch.LongTensor({1, 3, 2, 4, 6, 5, 7, 9, 8, 10, 12, 11, 13, 15, 14, 16, 18, 17, 20, 19, 22, 21, 24, 23})) -- swap Left/Right
         joints3D = joints3D:index(1, torch.LongTensor(opt.jointsIx)) 

         joints3D = joints3D:index(2, torch.LongTensor({3, 2, 1}))
         joints3D[{{}, {1}}] =  -joints3D[{{}, {1}}]
         joints3D[{{}, {3}}] =  -joints3D[{{}, {3}}]
      else print(path .. ' has weirdness (joints3D)' .. t); return nil end
   else
      if(opt.verbose) then print('Joints3D not loaded ' .. path) end
   end
   return joints3D
end

-- Joints3D (SMPL skeleton)
local function loadJoints3DSMPL(path, t)
   local joints3D, vars
   if pcall(function() vars = matio.load( getMatFile(path, '_info'), 'joints3D') end) then
      if pcall(function() joints3D = vars[{{}, {}, { t }}]:squeeze():t(); joints3D = joints3D:index(1, torch.LongTensor(opt.jointsIx)) end) then       -- [24 x 3]  
      else print(path .. ' has weirdness (joints3D)' .. t); return nil end
   else
      if(opt.verbose) then print('Joints3D not loaded ' .. path) end
   end
   return joints3D
end

-- SMPL pose parameters -- -- return 0 for now!
local function loadPose(path, t)
   poseRotVec = torch.zeros(3*24)
   poseRotMat = torch.zeros(9*24)

   if(opt.supervision == 'poseRotMat') then
      pose = poseRotMat
   elseif(opt.supervision == 'pose') then
      pose = poseRotVec
   end
   return pose
end

-- Segmentation
local function loadSegm(path, t)
   local segm
   if pcall(function() segm = matio.load( getMatFile(path, '_segm'), 'segm_' .. t) end) then -- [240 x 320]
   else
      if(opt.verbose) then print('Segm not loaded ' .. path) end;  return nil 
   end
   if(segm == nil) then; return nil; end
   local sIx = {2, 9, 12, 2, 10, 13, 2, 11, 14, 2, 11, 14, 2, 2, 2, 1, 3, 6, 4, 7, 5, 8, 5, 8}

   if(string.match(opt.supervision, 'segm')) then
      segm = changeSegmIx(segm, sIx) -- 0, 23/24? [240x320]
   else --if(string.match(opt.input, 'segm')) then -- including voxels supervision
      segm = changeSegmIx15(segm, sIx) -- [15x240x320]
   end

   return segm
end

-- Depth
local function loadDepth(path, t, dPelvis)-- Note dPelvis is output not input
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

local function loadVoxels(path, t)
   return torch.ByteTensor(opt.nVoxels, 45, 46):fill(0):contiguous()
end

M.getDuration      = getDuration
M.loadRGB          = loadRGB
M.loadJoints2D     = loadJoints2D
M.loadJoints3D     = loadJoints3D
M.loadJoints3DSMPL = loadJoints3DSMPL
M.loadPose         = loadPose
M.loadSegm         = loadSegm
M.loadDepth        = loadDepth
M.parseRGBPath     = parseRGBPath
M.loadVoxels       = loadVoxels

return M