paths.dofile('dataset.lua')
paths.dofile('util/camerautils.lua')
paths.dofile('util/eval.lua')
paths.dofile('util/img.lua')
paths.dofile('util/misc.lua')
paths.dofile('util/voxelutils.lua')
Rotations = paths.dofile('util/Rotations.lua')
require 'image'
matio = require 'matio'
cv = require 'cv'
require 'cv.videoio'

local zm
if opt.show then
   require 'qtwidget'
   wz = 1 -- global window zoom
   zm = wz*4 -- zoom for 1/4 images

   if opt.rgb then
      wr = qtwidget.newwindow(opt.sampleRes*wz, opt.sampleRes*wz, 'RGB')
   end
   if opt.depth then
      wd = qtwidget.newwindow(opt.sampleRes*wz, opt.sampleRes*wz, 'DEPTH')
   end
   if opt.segm then
      ws = qtwidget.newwindow(opt.sampleRes*wz, opt.sampleRes*wz, 'SEGM')
   end
   if opt.voxels then
      wv = qtwidget.newwindow(opt.nVoxels*2*4, opt.nVoxels*2, 'Voxels')
   end
   if opt.partvoxels then
      wvbp = qtwidget.newwindow(opt.nVoxels*2*4, opt.nVoxels*2, 'Voxels')
   end
end

-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------

-- a cache file of the training metadata (if doesnt exist, will be created)
local trainCache = paths.concat(opt.cache, opt.trainDir .. 'Cache.t7')
local testCache = paths.concat(opt.cache, opt.testDir .. 'Cache.t7')

-- Check for existence of opt.data
if not os.execute('cd ' .. opt.data) then
    error(("could not chdir to '%s'"):format(opt.data))
end

function changeSegmIx(segm, s)
   local out = torch.zeros(segm:size())
   for i = 1,#s do out[segm:eq(i)] = s[i] end
   return out
end

-- if s={1,2,...14} then it converts indices to one-hot
-- or it could be  {2, 12, 9, 2, 13, 10, 2, 14, 11, 2, 14, 11, 2, 2, 2, 1, 6, 3, 7, 4, 8, 5, 8, 5}
-- to merge 24 classes into 14
function changeSegmIx15(segm, s) -- public function -- move to utils
   local out = torch.zeros(opt.segmClasses, segm:size(1), segm:size(2))
   for i = 1,#s do
     out[s[i]+1][segm:eq(i)] = 1 
   end
   out[1] = out:sum(1):eq(0) -- background channel
   return out
end

function segm2index(segm)
   local out = torch.ones(segm:size(2), segm:size(3))
   for i = 1, segm:size(1) do
      out[segm[i]:byte()] = i
   end
   return out
end

-- voxels: e.g. tensor of size 128x128x128 with indices: 0, 1, 2, 3, 4, 5, 6, 7
-- s: e.g. {1,2,3,4,5,6}
function partvoxels2onehot(voxels, s)
   local out = torch.zeros(opt.nParts3D, opt.nVoxels, opt.nVoxels, opt.nVoxels)
   for i = 1,#s do
     out[s[i]+1][voxels:eq(i)] = 1 
   end
   out[1] = out:sum(1):eq(0) -- background channel
   return out
end

local loader

-- LOADING FUNCTIONS
if(opt.datasetname == 'H36M') then
   loader = paths.dofile('util/dataloader/H36M.lua')
elseif(opt.datasetname == 'cmu') then
   loader = paths.dofile('util/dataloader/SURREAL.lua')
elseif(opt.datasetname == 'UP') then
      loader = paths.dofile('util/dataloader/UP.lua')
else
   print('No loader file found for this dataset.')
end

--------------------------------------------------------------------------------
--[[
   Section 1: Create a train data loader (trainLoader),
   which does class-balanced sampling from the dataset and does a random crop
--]]

-- function common to do processing on loaded image/label
local Hook = function(self, path, set)
   collectgarbage()
   local rgbFull, rgb
   local camLoc
   local joints2D, joints3D
   local depth, depthFull, nForeground
   local segm, segmFull
   local pelvis3D
   local joints2Dhm, joints3Dhm
   local pose, shape
   local voxels, partvoxels
   local input, label
   local matPath
   local RT, R, T1
   local RT2, R2, T2
   local K, Kinv
   local j3d_c, j3d_c_a
   local cropinfo -- for UP cropped images from MPII

   local flip_prob = false
   if(set == 'train' and opt.hflip) then
      flip_prob = torch.uniform() < .5 -- do hflip with probability 0.5
   end

   local t1
   local iT

   if(opt.datasetname == 'H36M' and paths.extname(path) == 'jpg') then
      t1, matPath = loader.parseRGBPath(path)
   elseif(opt.datasetname == 'cmu' or opt.datasetname == 'H36M') then
      iT = loader.getDuration(path)
      if(set == 'train') then -- take random
         t1 = math.ceil(torch.uniform(1, iT))
      elseif(set == 'test') then -- take middle
         t1 = math.ceil(iT/2)
      end
   elseif(opt.datasetname == 'UP') then
      t1 = 1
   end

   -- load input
   if(opt.rgb) then
      rgbFull = loader.loadRGB(path, t1)
   end
   -- load 2D joints
   joints2D = loader.loadJoints2D(path, t1) -- [ 2 x nJoints]

   if(opt.joints3D) then
      joints3D = loader.loadJoints3D(path, t1) -- [ 3 x nJoints] 
      if(joints3D ~= nil) then
         -- if njoints == 16, elseif == 24 pelvis index=1
         pelvis3D = joints3D[7]:clone()
      end
   end

   -- Camera related
   if(opt.datasetname == 'cmu') then
      K = getIntrinsicBlender()
      Kinv = torch.inverse(K)
      camLoc = loader.loadCameraLoc(path) -- [3]
      RT, R, T1 = getExtrinsicBlender(camLoc:squeeze())
   elseif(opt.datasetname == 'H36M') then
      K = getIntrinsicH36M()
      Kinv = torch.inverse(K)
      camLoc = torch.zeros(3)
      RT, R, T1 = getExtrinsicBlender(camLoc:squeeze())
   elseif(opt.datasetname == 'UP') then
      local flength = loader.loadFocalLength(path, t)
      cropinfo = loader.getCropInfo(path, t)
      K = getIntrinsicUP(flength[1], cropinfo[2], cropinfo[1]) --cropinfow, cropinfoh
      Kinv = torch.inverse(K)
      camLoc = loader.loadCameraTrans(path, t)
      R = torch.Tensor({ {1, 0, 0}, {0, 1, 0}, {0, 0, 1} }) 
      T1 = camLoc:squeeze()
      RT = torch.cat(R, T1, 2)
   else
      print('Intrinsics&Extrinsics are not defined for this dataset.')
   end

   local dPelvis 
   -- depthFull and depth are quantized depths before and after cropping
   if(opt.depth and pelvis3D ~= nil) then
      if((opt.datasetname == 'cmu') and camLoc ~= nil) then
         -- depth of the pelvis in the depth image
         dPelvis = camLoc[1][1] - pelvis3D[1] 
      elseif(opt.datasetname == 'H36M') then
         joints3DSMPL = loader.loadJoints3DSMPL(path, t1) -- [ 3 x nJoints] 
         if(joints3DSMPL ~= nil) then
            pelvis3DSMPL = joints3DSMPL[7]:clone()
            dPelvis = pelvis3DSMPL[3]
         else
            print('joints3DSMPL is nil ' .. path)
         end
      else
         dPelvis = 0
      end
      depthFull, nForeground = loader.loadDepth(path, t1, dPelvis)
   end

   if(opt.joints3D and pelvis3D ~= nil) then
      -- Camera coordinates
      j3d_c = bsxfunsum((R*  joints3D:t()):t(),   T1) -- RT*[J3D; 1] -- xyz order (joints3D is zyx order)

      -- Align 3D joints positions so that pelvis is at [0, 0, 0]
      for jj = 1, #opt.jointsIx do
         joints3D[{{jj}, {}}] = joints3D[{{jj}, {}}]:squeeze() - pelvis3D
      end
      -- Aligned
      j3d_c_a = bsxfunsum(j3d_c, -j3d_c[7]:clone()) -- 7!! nJoints==24
   end
   -- segmFull and segm are segmentation masks before and after cropping
   if(opt.segm) then
      segmFull = loader.loadSegm(path, t1)
   end

   if(opt.pose) then
      pose = loader.loadPose(path, t1)
   end

   if(opt.shape) then
      shape = loader.loadShape(path, t1)
   end

   if(opt.voxels) then
      voxels = loader.loadVoxels(path, t1)
   end

   if(opt.partvoxels) then
      partvoxels = loader.loadPartVoxels(path, t1)
   end

   -- Don't do operations on the loaded variables until this point
   -- They might be nil
   if(   (opt.rgb        and rgbFull    == nil) 
      or (                   joints2D   == nil)  
      or (opt.depth      and depthFull  == nil)  
      or (opt.segm       and segmFull   == nil)  
      or (opt.joints3D   and joints3D   == nil) 
      or (opt.pose       and pose       == nil) 
      or (opt.voxels     and voxels     == nil) 
      or (opt.partvoxels and partvoxels == nil) 
      or (opt.shape      and shape == nil)
      ) then
         if(opt.verbose) then print('Nil! ' .. path) end
         return nil, nil, nil
   end

   local actualLoadSize 
   if(opt.rgb) then
      actualLoadSize = {1, rgbFull:size(2), rgbFull:size(3)}
   else
      actualLoadSize = {1, 240, 320}
   end

   -- Crop, scale
   local rot = 0
   local scale = getScale(joints2D, actualLoadSize[2] )
   local center = getCenter(joints2D)
   if (center[1] < 1 or center[2] < 1 or center[1] > actualLoadSize[3] or center[2] > actualLoadSize[2]) and (opt.datasetname == 'cmu') then
      if(opt.verbose) then print('Human out of image ' .. path .. ' center: ' .. center[1] .. ', ' .. center[2]) end
      return nil, nil, nil
   end

   -- Scale and rotation augmentation (randomly samples on a normal distribution)
   if(set == 'train') then
      local function rnd(x) return math.max(-2*x,math.min(2*x,torch.randn(1)[1]*x)) end
      scale = scale * (2 ^ rnd(opt.scale))
      rot = rnd(opt.rotate)
      if torch.uniform() <= .6 then rot = 0 end
   end

   -- ### CROP ###
   if(opt.rgb) then
      rgb = crop(rgbFull, center, scale, rot, opt.sampleRes, 'bilinear')
   end

   if(opt.supervision == 'joints2D'
      or opt.supervision == 'segm15joints2Djoints3D'
      or opt.supervision == 'segm15joints2Djoints3Dvoxels'
      or opt.supervision == 'segm15joints2Djoints3Dpartvoxels') then
      joints2Dhm = torch.zeros(#opt.jointsIx, opt.heatmapSize, opt.heatmapSize)
      for j = 1,#opt.jointsIx do
         if joints2D[{{j}, {2}}]:squeeze() > 1 then -- Checks that there is a ground truth annotation
            drawGaussian(joints2Dhm[j], transform(joints2D[{{j}, {}}]:squeeze(), center, scale, rot, opt.heatmapSize), 1) -- 1 is sigma
         end
      end
   elseif(opt.input == 'joints2D' 
      or opt.input == 'segm15joints2D'
      or opt.input == 'rgbsegm15joints2D'
      or opt.input == 'rgbsegm15joints2Djoints3D') then
      joints2Dhm = torch.zeros(#opt.jointsIx, opt.sampleRes, opt.sampleRes)
      for j = 1,#opt.jointsIx do
         if joints2D[{{j}, {2}}]:squeeze() > 1 then -- Checks that there is a ground truth annotation
            drawGaussian(joints2Dhm[j], transform(joints2D[{{j}, {}}]:squeeze(), center, scale, rot, opt.sampleRes), 1) -- 1 is sigma
         end
      end
   end

   if(opt.joints3D) then
      joints3Dhm = torch.zeros(#opt.jointsIx, opt.depthClasses, opt.heatmapSize, opt.heatmapSize) -- 16 x 19 x 64 x 64
      for j = 1,#opt.jointsIx do
         if joints2D[{{j}, {2}}]:squeeze() > 1 then -- Checks that there is a ground truth annotation
            local t_joints2D = transform(joints2D[{{j}, {}}]:squeeze(), center, scale, rot, opt.heatmapSize)
            local x = t_joints2D[1] -- x from joints2D
            local y = t_joints2D[2] -- y from joints2D
            local z
            if(opt.datasetname == 'UP') then
               z = -joints3D[{{j}, {3}}]:squeeze() -- z from joints3D
            else
               z = joints3D[{{j}, {1}}]:squeeze() -- z from joints3D
            end

            local lowB = -(opt.depthClasses - 1)/2
            local upB = (opt.depthClasses - 1)/2

            z = math.max(math.min(torch.round(z/opt.stp), upB), lowB) -- quantize (already aligned)
            z = z + 1+upB -- so that it's between 1-19. It was [-9 .. 0 .. 9].
            draw3DGaussian(joints3Dhm[j], {z, y, x}, 1) -- z, y, x order, because we want z to be #channels 1 is for sigma
         end
      end

      local j3d_q_64 = getPreds3DCropped(joints3Dhm:view(#opt.jointsIx * opt.depthClasses, opt.heatmapSize, opt.heatmapSize), opt.depthClasses)
   
      j3d_q_64 = j3d_q_64:index(2, torch.LongTensor({3, 2, 1})) -- zyx -> xyz
      -- j2d_q is the coarse xy image coordinates before cropping 
      local j2d_q = torch.Tensor(#opt.jointsIx, 2)
      for j = 1,j3d_q_64:size(1) do
         j2d_q[j] = transform(j3d_q_64[{{j}, {1, 2}}]:squeeze(),center,scale,rot,opt.heatmapSize,true)
      end

      -- Using quantized z
      local middleZ = (opt.depthClasses + 1)/2
      -- Camera related
      if(opt.datasetname == 'cmu' or opt.datasetname == 'H36M') then
         RT2, R2, T2 = getExtrinsicBlender(pelvis3D)
      elseif(opt.datasetname == 'UP') then
         --RT, R, T1 = getExtrinsicBlender(camLoc:squeeze())
         R2 = torch.Tensor({ {1, 0, 0}, {0, 1, 0}, {0, 0, 1} }) 
         T2 = pelvis3D
         RT2 = torch.cat(R2, T2, 2)
      else
         print('Intrinsics&Extrinsics are not defined for this dataset.')
      end

      -- Won't work for UP
      local rec3D_q = reconstruct3D(j2d_q-1, Kinv, bsxfunsum(-(j3d_q_64 - middleZ)*opt.stp, T1-T2))-- - R*joints3D[7]))
      -- Align
      rec3D_q = bsxfunsum(rec3D_q, -rec3D_q[7]:clone())
   end

   if(opt.depth) then
      depth = crop(depthFull, center, scale, rot, opt.heatmapSize, 'simple')
      if(opt.supervision == 'depth') then -- classification
         depth = depth + 1 -- it was kept [0-19] until cropping because it puts zero when rotating.
      -- we don't want 0 because they are class indices (should be positive)
      end 
      if opt.show then
         print('Range [' .. depth:min() .. ', ' .. depth:max() .. ']')
         if(opt.supervision == 'depth') then
            depth[{{1}, {1}}] = 1
            depth[{{1}, {2}}] = opt.depthClasses + 1
            wd = image.display({image=image.y2jet(depth), win=wd, zoom=zm})
         end
      end
   end

   if(opt.segm) then
      if(string.match(opt.supervision, 'segm')) then
         segm = crop(segmFull, center, scale, rot, opt.heatmapSize, 'simple')
      else --if(string.match(opt.input, 'segm')) then -- voxels supervision
         segm = crop(segmFull, center, scale, rot, opt.sampleRes, 'simple')
      end
      if(string.match(opt.supervision , 'segm')) then
         segm = segm + 1 -- same trick as in depth
      end
   end

   local segmFV, segmSV
   local silhFV, silhSV
   if(opt.voxels or opt.partvoxels) then
      if rot ~= 0 then print('Rot should be 0!') end

      -- TODO can be optimized, it always relies on 15 channel
      if(segmFull:size(1) == opt.segmClasses) then
         local segmtemp = crop(segmFull, center, scale, rot, opt.nVoxels, 'simple')
         silhFV = segmtemp[{{2, -1}, {}, {}}]:sum(1):squeeze():gt(0) -- why not segmtemp[1]

         if(opt.datasetname == 'UP') then
         -- On UP dataset, there are two types of segmentation, one hand labelled silhouette
         -- and other coming from the SMPL fits. For 2D segmentation that requires 15 parts,
         -- we use 31 parts from the SMPL fits combined into 15. For projection, we experiment
         -- with using the silhouette or the 31
            local silhtemp = loader.loadSilhouette(path, t)
            silhFV = crop(silhtemp:float(), center, scale, rot, opt.nVoxels, 'simple')
         end

      else --if(segmFull:size(1) == 1 or 64) then
         local six = {}
         for pid = 1, opt.segmClasses-1 do
            table.insert(six, pid)
         end
         segmFull2 = changeSegmIx15(segmFull, six)
         local segmtemp = crop(segmFull2, center, scale, rot, opt.nVoxels, 'simple')
         silhFV = segmtemp[{{2, -1}, {}, {}}]:sum(1):squeeze():gt(0) -- why not segmtemp[1]
      end
      
      if(silhFV:sum(1):nonzero():dim() == 0) then
         print('Silhouette all zeros!')
         return nil, nil, nil
      end

   end

   local trans_params
   if(opt.voxels) then
      voxels, trans_params = alignGTVoxels(voxels, silhFV, joints2D, path)
      voxels = voxels:permute(3, 1, 2)
      voxels = image.flip(voxels:contiguous(), 1)

      if opt.show then
         local voximgs = showVoxels(voxels:float())
         image.display({image={silhFV:repeatTensor(3, 1, 1)*voximgs[3]:max(), voximgs[1], voximgs[2], voximgs[3]}, win=wv, zoom=2})
      end
   end

   if(opt.partvoxels) then
      partvoxels, trans_params = alignGTVoxels(partvoxels, silhFV, joints2D, path)
      partvoxels:add(1)

      partvoxels = partvoxels:permute(3, 1, 2)
      partvoxels = image.flip(partvoxels:contiguous(), 1)

      if opt.show then
         local voximgs = showPartVoxels(partvoxels)
         -- TO-DO show segmFV
         --image.display({image={image.y2jet(segmFV), voximgs[1], voximgs[2], voximgs[3]}, win=wvbp, zoom=2})
         image.display({image=voximgs, win=wvbp, zoom=2})
      end
   end

   if(opt.proj == 'silhFVSV') then
      silhSV = voxels:max(3):squeeze()
   end

   if(opt.proj == 'segmFV' or opt.proj == 'segmFVSV') then
      local partvoxelsonehot = partvoxels2onehot(torch.add(partvoxels, -1), {1,2,3,4,5,6})
      segmFV = torch.zeros(opt.nParts3D, opt.nVoxels, opt.nVoxels)
      segmFV[{{2, opt.nParts3D}, {}, {}}] = partvoxelsonehot[{{2, opt.nParts3D}, {}, {}, {}}]:max(2):squeeze()
      segmFV[1] = partvoxelsonehot[1]:min(1):squeeze()

   end

   if(opt.proj == 'segmFVSV') then
      local partvoxelsonehot = partvoxels2onehot(torch.add(partvoxels, -1), {1,2,3,4,5,6}) 
      segmSV = torch.zeros(opt.nParts3D, opt.nVoxels, opt.nVoxels)
      segmSV[{{2, opt.nParts3D}, {}, {}}] = partvoxelsonehot[{{2, opt.nParts3D}, {}, {}, {}}]:max(4):squeeze()
      segmSV[1] = partvoxelsonehot[1]:min(3):squeeze()
   end

   if(set == 'train') then
      if flip_prob then
         if(opt.rgb) then rgb = flip(rgb) end
         if(joints2Dhm ~= nil) then joints2Dhm = shuffleLRJoints(flip(joints2Dhm)); end
         if(opt.joints3D) then joints3Dhm = shuffleLRJoints(flip(joints3Dhm)); end
         if(opt.segm) then
            if(segm:nDimension() == 3) then
               segm = shuffleLRSegm(flip(segm)) ;
            else
               segm = shuffleLRSegm(image.hflip(segm))
            end
         end
         if(opt.depth) then depth = flip(depth); print('Depth flip not checked.')  end
         if(opt.voxels) then voxels = flip(voxels); print('Voxels flip not checked.') end
      end
      
      -- Color augmentation
      if(opt.rgb) then
        for c=1, 3 do
            local rndJitter = torch.uniform(0.6,1.4)
            rgb[{{c}, {}, {}}]:mul(rndJitter):clamp(0,1)
        end
      end
   end


   if opt.show then

      local preds_joints2D, preds_img, im
      if(opt.supervision == 'joints2D') then
         preds_joints2D, preds_img = getPreds2D(joints2Dhm, center, scale)
      end
      --local im
      if(opt.rgb) then  
         if(opt.supervision == 'joints2D') then       
            im = drawSkeleton(rgb[{{}, {}, {}}], joints2Dhm, preds_joints2D[1]*4)
         else
            im = rgb[{{}, {}, {}}]
         end
         wr = image.display({image=im, win=wr, zoom=wz})
         if(opt.supervision == 'joints2D') then
            for j = 1,#opt.jointsIx do
               wr:setcolor(0,0,1)
               wr:arc(4*preds_joints2D[{{1}, {j}, {1}}]:squeeze()*wz, 4*preds_joints2D[{{1}, {j}, {2}}]:squeeze()*wz, wz, 0, 360)
               wr:stroke()
            end
         end
         sys.sleep(0.1)
      end
      if(opt.segm) then
         local segmImg
         if(segm:size(1) == opt.segmClasses) then
         --if(opt.input == 'segm15' 
         --   or opt.input == 'rgbsegm15joints2D' or opt.input == 'rgbsegm15joints2Djoints3D'
         --   or opt.supervision == 'voxels') then
            ws = image.display({image=segm15Img(segm), win=ws, zoom=wz})
         else
            segmImg = segm
            ws = image.display({image=image.y2jet(segmImg), win=ws, zoom=wz*4})
         end
      end
   end

   if(opt.input == 'rgb') then
      input = rgb
   elseif(opt.input == 'segm15') then
      input = segm
   elseif(opt.input == 'joints2D') then
      input = joints2Dhm
   elseif(opt.input == 'depth') then
      input = depth
   elseif(opt.input == 'segm15joints2D') then
      input = torch.cat(segm, joints2Dhm, 1)
   elseif(opt.input == 'rgbsegm15joints2D') then
      input = torch.cat({rgb, segm, joints2Dhm}, 1)
   elseif(opt.input == 'rgbsegm15joints2Djoints3D') then
      input = {torch.cat({rgb, segm, joints2Dhm}, 1), joints3Dhm:view(#opt.jointsIx * opt.depthClasses, opt.heatmapSize, opt.heatmapSize)}
   elseif(opt.input == 'joints3D') then
      input = joints3Dhm:view(#opt.jointsIx * opt.depthClasses, opt.heatmapSize, opt.heatmapSize)
   elseif(opt.input == 'segm15joints3D') then
      input = {segm:contiguous(), joints3Dhm:view(#opt.jointsIx * opt.depthClasses, opt.heatmapSize, opt.heatmapSize)}
   else
      error('Input type not supported.')
   end

   if(opt.whiten) then
      local ch = #opt.mean
      for c = 1, ch do
         if opt.mean then input[{{c}, {}, {}}]:add(-opt.mean[c]) end
         if  opt.std then input[{{c}, {}, {}}]:div(opt.std[c]) end
      end
   end

   if(opt.supervision == 'depth') then
      label = depth
   elseif(opt.supervision == 'segm') then
      label = segm
   elseif(opt.supervision == 'joints2D') then
      label = joints2Dhm
   elseif(opt.supervision == 'joints3D') then
      label = joints3Dhm:view(#opt.jointsIx * opt.depthClasses, opt.heatmapSize, opt.heatmapSize)
   elseif(opt.supervision == 'pose' or opt.supervision == 'poseRotMat') then
      label = pose:view(-1, 1, 1)
   elseif(opt.supervision == 'shape') then
      label = shape:view(10, 1, 1)
   elseif(opt.supervision == 'poseshape') then
      label = torch.cat(pose:view(-1, 1, 1), shape:view(10, 1, 1), 1)
   elseif(opt.supervision == 'voxels') then
      label = voxels
   elseif(opt.supervision == 'partvoxels') then
      label = partvoxels
   elseif(opt.supervision == 'segm15joints2Djoints3D') then
      label = {segm, joints2Dhm, joints3Dhm:view(#opt.jointsIx * opt.depthClasses, opt.heatmapSize, opt.heatmapSize)}
   elseif(opt.supervision == 'segm15joints2Djoints3Dvoxels') then
      label = {segm, joints2Dhm, joints3Dhm:view(#opt.jointsIx * opt.depthClasses, opt.heatmapSize, opt.heatmapSize), voxels}
   elseif(opt.supervision == 'segm15joints2Djoints3Dpartvoxels') then
      label = {segm, joints2Dhm, joints3Dhm:view(#opt.jointsIx * opt.depthClasses, opt.heatmapSize, opt.heatmapSize), partvoxels} 
   else
      error('Supervision type not supported.')
   end

   local instance = {}
   instance.depth = depth
   instance.segm = segm
   instance.joints2Dhm = joints2Dhm
   instance.joints3Dhm = joints3Dhm
   instance.joints3D = joints3D
   instance.joints2D = joints2D
   instance.pose = pose
   instance.shape = shape
   instance.voxels = voxels
   instance.partvoxels = partvoxels
   instance.t = t1
   instance.path = path
   instance.camLoc = camLoc
   instance.pelvis3D = pelvis3D
   instance.j3d_c = j3d_c -- joints3D in camera coordinates
   instance.j3d_c_a = j3d_c_a -- joints3D aligned to pelvis
   instance.segmFV = segmFV
   instance.silhFV = silhFV
   instance.segmSV = segmSV
   instance.silhSV = silhSV
   instance.K = K
   instance.R = R
   instance.T1 = T1
   instance.RT = RT
   instance.R2 = R2
   instance.T2 = T2
   instance.RT2 = RT2
   instance.cropinfo = cropinfo

   collectgarbage()
   return input, label, instance
end


-- function to load the train image
trainHook = function(self, path)
   return Hook(self, path, 'train')
end

if paths.filep(trainCache) then
   --print('Loading train metadata from cache')
   trainLoader = torch.load(trainCache)

   -- TEMPORARY 
   if opt.protocol == 'p1' then
      local cache = torch.load('/home/gvarol/cnn_saves/H36M/cache/train_p1Cache.t7')

      trainLoader.classListSample = cache.classListSample
      trainLoader.paths = cache.paths
      trainLoader.numSamples = cache.numSamples
      trainLoader.imagePath = cache.imagePath
      trainLoader.imageClass = cache.imageClass
      trainLoader.classList = cache.classList
      print(trainLoader.numSamples .. ' SAMPLES.')
   -- TEMPORARY
   end

   trainLoader.sampleHookTrain = trainHook
   --assert(trainLoader.paths[1] == paths.concat(opt.data, 'train'),
   --       'cached files dont have the same path as opt.data. Remove your cached files at: '
   --          .. trainCache .. ' and rerun the program')
else
   print('Creating train metadata')
   trainLoader = dataLoader{
      paths = {paths.concat(opt.data, opt.trainDir)},
      split = 100,
      verbose = true,
      forceClasses = opt.forceClasses
   }
   torch.save(trainCache, trainLoader)
   trainLoader.sampleHookTrain = trainHook
end
collectgarbage()

-- do some sanity checks on trainLoader
do
   local class = trainLoader.imageClass
   local nClasses = #trainLoader.classes
   assert(class:max() <= nClasses, "class logic has error")
   assert(class:min() >= 1, "class logic has error")

end

-- function to load the test image
testHook = function(self, path)
   return Hook(self, path, 'test')
end

if paths.filep(testCache) then
   --print('Loading test metadata from cache')
   testLoader = torch.load(testCache)
   testLoader.sampleHookTest = testHook
   assert(testLoader.paths[1] == paths.concat(opt.data, opt.testDir),
          'cached files dont have the same path as opt.data. Remove your cached files at: '
             .. testCache .. ' and rerun the program')
else
   print('Creating test metadata')
   print('Test dir: ' .. opt.testDir)
   testLoader = dataLoader{
      paths = {paths.concat(opt.data, opt.testDir)},
      split = 0,
      verbose = true,
      forceClasses = trainLoader.classes -- force consistent class indices between trainLoader and testLoader
   }
   torch.save(testCache, testLoader)
   testLoader.sampleHookTest = testHook
end
collectgarbage()
-- End of test loader section