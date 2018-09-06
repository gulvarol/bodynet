require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')
local ffi = require 'ffi'
local class = require('pl.class')
local dir = require 'pl.dir'
local tablex = require 'pl.tablex'
local argcheck = require 'argcheck'
require 'sys'
require 'xlua'
require 'image'

local dataset = torch.class('dataLoader')

local initcheck = argcheck{
   pack=true,
   help=[[
     A dataset class for images in a flat folder structure (folder-name is class-name).
     Optimized for extremely large datasets (upwards of 14 million images).
     Tested only on Linux (as it uses command-line linux utilities to scale up)
]],
   {check=function(paths)
       local out = true;
       for k,v in ipairs(paths) do
          if type(v) ~= 'string' then
             print('paths can only be of string input');
             out = false
          end
       end
       return out
   end,
    name="paths",
    type="table",
    help="Multiple paths of directories with images"},

   {name="split",
    type="number",
    help="Percentage of split to go to Training"
   },

   {name="samplingMode",
    type="string",
    help="Sampling mode: random | balanced ",
    default = "balanced"},

   {name="verbose",
    type="boolean",
    help="Verbose mode during initialization",
    default = false},

   {name="forceClasses",
    type="table",
    help="If you want this loader to map certain classes to certain indices, "
       .. "pass a classes table that has {classname : classindex} pairs."
       .. " For example: {3 : 'dog', 5 : 'cat'}"
       .. "This function is very useful when you want two loaders to have the same "
    .. "class indices (trainLoader/testLoader for example)",
    opt = true},

   {name="sampleHookTrain",
    type="function",
    help="applied to sample during training(ex: for lighting jitter). "
       .. "It takes the image path as input",
    opt = true},

   {name="sampleHookTest",
    type="function",
    help="applied to sample during testing",
    opt = true},
}

function dataset:__init(...)

   -- argcheck
   local args =  initcheck(...)
   print(args)
   for k,v in pairs(args) do self[k] = v end

   -- find class names
   self.classes = {}
   local classPaths = {}
   if self.forceClasses then
      for k,v in pairs(self.forceClasses) do
         self.classes[k] = v
         classPaths[k] = {}
      end
   end
   local function tableFind(t, o) for k,v in pairs(t) do if v == o then return k end end end
   -- loop over each paths folder, get list of unique class names,
   -- also store the directory paths per class
   -- for each class,
   for k,path in ipairs(self.paths) do -- paths = {<fullpath>/train/}
      local dirs = dir.getdirectories(path); 
      for k,dirpath in ipairs(dirs) do -- dirs = {01_01, 01_02, ...}
         local class = paths.basename(dirpath)
         local idx = tableFind(self.classes, class)
         if not idx then
            table.insert(self.classes, class) -- class = 'run0', 'run1', 'run2'
            idx = #self.classes
            classPaths[idx] = {}
         end
         if not tableFind(classPaths[idx], dirpath) then
            table.insert(classPaths[idx], dirpath);
         end
      end
   end

   self.classIndices = {} -- classIndices = 'run0' -> 3, 'run1' -> 1, 'run2' -> 2
   for k,v in ipairs(self.classes) do
      self.classIndices[v] = k
   end

   -- define command-line tools, try your best to maintain OSX compatibility
   local wc = 'wc'
   local cut = 'cut'
   local find = 'find'
   if jit.os == 'OSX' then
      wc = 'gwc'
      cut = 'gcut'
      find = 'gfind'
   end
   ----------------------------------------------------------------------
   -- Options for the GNU find command
   local extensionList = opt.extension -- {'mp4'} --{'jpg'}

   -- Following for H36M val so that mp4 folders don't get on the list
   --local findOptions = ' -maxdepth 2 -type f -iname "*.' .. extensionList[1] .. '"'
   local findOptions = ' -type f -iname "*.' .. extensionList[1] .. '"'
   for i=2,#extensionList do
      findOptions = findOptions .. ' -o -iname "*.' .. extensionList[i] .. '"'
   end

   -- find the image path names
   self.imagePath = torch.CharTensor()  -- path to each image in dataset
   self.imageClass = torch.LongTensor() -- class index of each image (class index in self.classes)
   self.classList = {}                  -- index of imageList to each image of a particular class
   self.classListSample = self.classList -- the main list used when sampling data -- same as classList.    1 : LongTensor - size: 25754 --2 : LongTensor - size: 12684 --3 : LongTensor - size: 16563 going from 1 to classSize

   print('running "find" on each class directory, and concatenate all'
         .. ' those filenames into a single file containing all image paths for a given class')
   -- so, generates one file per class
   local classFindFiles = {}
   for i=1,#self.classes do
      classFindFiles[i] = os.tmpname()
   end
   local combinedFindList = os.tmpname();

   local tmpfile = os.tmpname()
   local tmphandle = assert(io.open(tmpfile, 'w'))
   -- iterate over classes
   for i, class in ipairs(self.classes) do
      -- iterate over classPaths
      for j,path in ipairs(classPaths[i]) do
         local command = find .. ' "' .. path .. '" ' .. findOptions
            .. ' >>"' .. classFindFiles[i] .. '" \n'
            print(command)
         tmphandle:write(command)
      end
   end
   io.close(tmphandle)
   os.execute('bash ' .. tmpfile)
   os.execute('rm -f ' .. tmpfile)

   print('now combine all the files to a single large file')
   local tmpfile = os.tmpname()
   local tmphandle = assert(io.open(tmpfile, 'w'))
   -- concat all finds to a single large file in the order of self.classes
   for i=1,#self.classes do
      local command = 'cat "' .. classFindFiles[i] .. '" >>' .. combinedFindList .. ' \n'
      tmphandle:write(command)
   end
   io.close(tmphandle)
   os.execute('bash ' .. tmpfile)
   os.execute('rm -f ' .. tmpfile)

   --==========================================================================
   print('load the large concatenated list of sample paths to self.imagePath')
   local maxPathLength = tonumber(sys.fexecute(wc .. " -L '"
                                                  .. combinedFindList .. "' |"
                                                  .. cut .. " -f1 -d' '")) + 1
   local length = tonumber(sys.fexecute(wc .. " -l '"
                                           .. combinedFindList .. "' |"
                                           .. cut .. " -f1 -d' '"))
   assert(length > 0, "Could not find any image file in the given input paths")
   assert(maxPathLength > 0, "paths of files are length 0?")
   self.imagePath:resize(length, maxPathLength):fill(0)
   local s_data = self.imagePath:data()
   local count = 0
   for line in io.lines(combinedFindList) do
      ffi.copy(s_data, line)
      s_data = s_data + maxPathLength
      if self.verbose and count % 10000 == 0 then
         xlua.progress(count, length)
      end;
      count = count + 1
   end

   self.numSamples = self.imagePath:size(1)
   if self.verbose then print(self.numSamples ..  ' samples found.') end
   --==========================================================================
   print('Updating classList and imageClass appropriately')
   self.imageClass:resize(self.numSamples)
   local runningIndex = 0
   for i=1,#self.classes do
      if self.verbose then xlua.progress(i, #(self.classes)) end
      local length = tonumber(sys.fexecute(wc .. " -l '"
                                              .. classFindFiles[i] .. "' |"
                                              .. cut .. " -f1 -d' '"))
      if length == 0 then
         error('Class has zero samples')
      else
         self.classList[i] = torch.linspace(runningIndex + 1, runningIndex + length, length):long()
         self.imageClass[{{runningIndex + 1, runningIndex + length}}]:fill(i)
      end
      runningIndex = runningIndex + length
   end

   --==========================================================================
   -- clean up temporary files
   print('Cleaning up temporary files')
   local tmpfilelistall = ''
   for i=1,#(classFindFiles) do
      tmpfilelistall = tmpfilelistall .. ' "' .. classFindFiles[i] .. '"'
      if i % 1000 == 0 then
         os.execute('rm -f ' .. tmpfilelistall)
         tmpfilelistall = ''
      end
   end
   os.execute('rm -f '  .. tmpfilelistall)
   os.execute('rm -f "' .. combinedFindList .. '"')
   --==========================================================================

   if self.split == 100 then
      self.testIndicesSize = 0
   else
      print('Splitting training and test sets to a ratio of '
               .. self.split .. '/' .. (100-self.split))
      self.classListTrain = {}
      self.classListTest  = {}
      self.classListSample = self.classListTrain
      local totalTestSamples = 0
      -- split the classList into classListTrain and classListTest
      for i=1,#self.classes do
         local list = self.classList[i]
         local count = self.classList[i]:size(1)
         local splitidx = math.floor((count * self.split / 100) + 0.5) -- +round
         local perm = torch.randperm(count)
         self.classListTrain[i] = torch.LongTensor(splitidx)
         for j=1,splitidx do
            self.classListTrain[i][j] = list[perm[j]]
         end
         if splitidx == count then -- all samples were allocated to train set
            self.classListTest[i]  = torch.LongTensor()
         else
            self.classListTest[i]  = torch.LongTensor(count-splitidx)
            totalTestSamples = totalTestSamples + self.classListTest[i]:size(1)
            local idx = 1
            for j=splitidx+1,count do
               self.classListTest[i][idx] = list[perm[j]]
               idx = idx + 1
            end
         end
      end
      -- Now combine classListTest into a single tensor
      self.testIndices = torch.LongTensor(totalTestSamples)
      self.testIndicesSize = totalTestSamples
      local tdata = self.testIndices:data()
      local tidx = 0
      for i=1,#self.classes do
         local list = self.classListTest[i]
         if list:dim() ~= 0 then
            local ldata = list:data()
            for j=0,list:size(1)-1 do
               tdata[tidx] = ldata[j]
               tidx = tidx + 1
            end
         end
      end
   end
end

-- size(), size(class)
function dataset:size(class, list)
    list = list or self.classList
    if not class then
        return self.numSamples
    elseif type(class) == 'string' then
        return list[self.classIndices[class]]:size(1)
    elseif type(class) == 'number' then
        return list[class]:size(1)
    end
end

-- getByClass
function dataset:getByClass(class)
    local classindex = math.ceil(torch.uniform() * self.classListSample[class]:nElement())
    local index = self.classListSample[class][classindex]
    local imgpath = ffi.string(torch.data(self.imagePath[index]))
    local input, label, instance = self:sampleHookTrain(imgpath)
    return input, label, instance, index
end

-- sampler, samples from the training set.
function dataset:sample(quantity)
    assert(quantity)
    local inputs, labels, instances, indices

    if(opt.mix) then
        labels = torch.Tensor(quantity, unpack(opt.outSize))
        indices = torch.Tensor(quantity)
        instances = {}
        -- Split the batch into two randomly
        -- First part for rgb input (will be transformed to predicted [segm] input)
        -- Second part for ground truth [segm] input
        local q1 = torch.random(quantity)
        local q2 = quantity - q1

        local inputs1, inputs2

        if(opt.applyHG == 'segm15') then
            inputs1 = torch.Tensor(q1, 3, opt.inSize[2], opt.inSize[3]) --
            inputs2 = torch.Tensor(q2, 15, opt.inSize[2], opt.inSize[3]) --

            opt.input = 'rgb'
            opt.inSize = {3, opt.sampleRes, opt.sampleRes}
            opt.rgb = true
            if(not opt.supervision == 'voxels') then
              opt.segm = false
            end
            opt.whiten = true

            local cnt = 1
            for i=1,q1 do
                local class = torch.random(1, #self.classes)
                local input, label, instance, index = self:getByClass(class)
                while (input == nil) do -- if the sample is nil for some reason (not read properly, size mismatch etc.) sample another one
                    input, label, instance, index = self:getByClass(class) -- !!
                end
                inputs1[i]:copy(input)
                labels[cnt]:copy(label)
                table.insert(instances, instance)
                indices[cnt] = index
                cnt = cnt + 1
             end

            opt.input = 'segm15'
            opt.inSize = {15, opt.sampleRes, opt.sampleRes}
            opt.rgb = false
            --opt.segm = true
            opt.whiten = false

            for i=1,q2 do
                local class = torch.random(1, #self.classes)
                local input, label, instance, index = self:getByClass(class)
                while (input == nil) do -- if the sample is nil for some reason (not read properly, size mismatch etc.) sample another one
                    input, label, instance, index = self:getByClass(class) -- !!
                end
                inputs2[i]:copy(input)
                labels[cnt]:copy(label)
                table.insert(instances, instance)
                indices[cnt] = index
                cnt = cnt + 1
             end

            -- For the validation epoch, put back to rgb input mode
            opt.input = 'rgb'
            opt.inSize = {3, opt.sampleRes, opt.sampleRes}
            opt.rgb = true
            if(not opt.supervision == 'voxels') then
              opt.segm = false
            end
            opt.whiten = true

        elseif(opt.applyHG == 'joints2D') then
            inputs1 = torch.Tensor(q1, 3, opt.inSize[2], opt.inSize[3]) --
            inputs2 = torch.Tensor(q2, #opt.jointsIx, opt.inSize[2], opt.inSize[3]) --

            opt.input = 'rgb'
            opt.inSize = {3, opt.sampleRes, opt.sampleRes}
            opt.rgb = true
            opt.whiten = true

            local cnt = 1
            for i=1,q1 do
                local class = torch.random(1, #self.classes)
                local input, label, instance, index = self:getByClass(class)
                while (input == nil) do -- if the sample is nil for some reason (not read properly, size mismatch etc.) sample another one
                    input, label, instance, index = self:getByClass(class) -- !!
                end
                inputs1[i]:copy(input)
                labels[cnt]:copy(label)
                table.insert(instances, instance)
                indices[cnt] = index
                cnt = cnt + 1
             end

            opt.input = 'joints2D'
            opt.inSize = {#opt.jointsIx, opt.sampleRes, opt.sampleRes}
            opt.rgb = false
            opt.whiten = false

            for i=1,q2 do
                local class = torch.random(1, #self.classes)
                local input, label, instance, index = self:getByClass(class)
                while (input == nil) do -- if the sample is nil for some reason (not read properly, size mismatch etc.) sample another one
                    input, label, instance, index = self:getByClass(class) -- !!
                end
                inputs2[i]:copy(input)
                labels[cnt]:copy(label)
                table.insert(instances, instance)
                indices[cnt] = index
                cnt = cnt + 1
             end

            -- For the validation epoch, put back to rgb input mode
            opt.input = 'rgb'
            opt.inSize = {3, opt.sampleRes, opt.sampleRes}
            opt.rgb = true
            opt.whiten = true
        elseif(opt.applyHG == 'segm15joints2D') then
            inputs1 = torch.Tensor(q1, 3, opt.inSize[2], opt.inSize[3]) --
            inputs2 = torch.Tensor(q2, (15 + #opt.jointsIx), opt.inSize[2], opt.inSize[3]) --

            opt.input = 'rgb'
            opt.inSize = {3, opt.sampleRes, opt.sampleRes}
            if(not opt.supervision == 'voxels') then
              opt.segm = false
            end
            opt.rgb = true
            opt.whiten = true

            local cnt = 1
            for i=1,q1 do
                local class = torch.random(1, #self.classes)
                local input, label, instance, index = self:getByClass(class)
                while (input == nil) do -- if the sample is nil for some reason (not read properly, size mismatch etc.) sample another one
                    input, label, instance, index = self:getByClass(class) -- !!
                end
                inputs1[i]:copy(input)
                labels[cnt]:copy(label)
                table.insert(instances, instance)
                indices[cnt] = index
                cnt = cnt + 1
             end

            opt.input = 'segm15joints2D'
            opt.inSize = {(15 + #opt.jointsIx), opt.sampleRes, opt.sampleRes}
            opt.segm = true
            opt.rgb = false
            opt.whiten = false

            for i=1,q2 do
                local class = torch.random(1, #self.classes)
                local input, label, instance, index = self:getByClass(class)
                while (input == nil) do -- if the sample is nil for some reason (not read properly, size mismatch etc.) sample another one
                    input, label, instance, index = self:getByClass(class) -- !!
                end
                inputs2[i]:copy(input)
                labels[cnt]:copy(label)
                table.insert(instances, instance)
                indices[cnt] = index
                cnt = cnt + 1
             end

            -- For the validation epoch, put back to rgb input mode
            opt.input = 'rgb'
            opt.inSize = {3, opt.sampleRes, opt.sampleRes}
            opt.rgb = true            
            if(not opt.supervision == 'voxels') then
              opt.segm = false
            end
            opt.whiten = true
        elseif(opt.applyHG == 'rgbsegm15joints2D') then
            inputs1 = torch.Tensor(q1, 3, opt.inSize[2], opt.inSize[3]) --
            inputs2 = torch.Tensor(q2, (3+15 + #opt.jointsIx), opt.inSize[2], opt.inSize[3]) --

            opt.input = 'rgb'
            opt.inSize = {3, opt.sampleRes, opt.sampleRes}
            if(not opt.supervision == 'voxels') then
              opt.segm = false
            end
            opt.rgb = true
            opt.whiten = true -- check!

            local cnt = 1
            for i=1,q1 do
                local class = torch.random(1, #self.classes)
                local input, label, instance, index = self:getByClass(class)
                while (input == nil) do -- if the sample is nil for some reason (not read properly, size mismatch etc.) sample another one
                    input, label, instance, index = self:getByClass(class) -- !!
                end
                inputs1[i]:copy(input)
                labels[cnt]:copy(label)
                table.insert(instances, instance)
                indices[cnt] = index
                cnt = cnt + 1
             end

            opt.input = 'rgbsegm15joints2D'
            opt.inSize = {(3 + 15 + #opt.jointsIx), opt.sampleRes, opt.sampleRes}
            opt.segm = true
            opt.rgb = true
            opt.whiten = false

            for i=1,q2 do
                local class = torch.random(1, #self.classes)
                local input, label, instance, index = self:getByClass(class)
                while (input == nil) do -- if the sample is nil for some reason (not read properly, size mismatch etc.) sample another one
                    input, label, instance, index = self:getByClass(class) -- !!
                end
                inputs2[i]:copy(input)
                labels[cnt]:copy(label)
                table.insert(instances, instance)
                indices[cnt] = index
                cnt = cnt + 1
             end

            -- For the validation epoch, put back to rgb input mode
            opt.input = 'rgb'
            opt.inSize = {3, opt.sampleRes, opt.sampleRes}
            opt.rgb = true
            if(not opt.supervision == 'voxels') then
              opt.segm = false
            end
            opt.whiten = true
        elseif(opt.applyHG == 'rgbsegm15joints2Djoints3D') then
            inputs1 = torch.Tensor(q1, 3, opt.inSize[2], opt.inSize[3]) --
            inputs2 = {} -- torch.Tensor(q2, (3+15 + #opt.jointsIx), opt.inSize[2], opt.inSize[3]) --
            table.insert(inputs2, torch.Tensor(q2, (3+15 + #opt.jointsIx), opt.inSize[2], opt.inSize[3]))
            table.insert(inputs2, torch.Tensor(q2, opt.depthClasses*#opt.jointsIx, opt.heatmapSize, opt.heatmapSize))

            opt.input = 'rgb'
            opt.inSize = {3, opt.sampleRes, opt.sampleRes}
            opt.rgb = true
            opt.whiten = true

            local cnt = 1
            for i=1,q1 do
                local class = torch.random(1, #self.classes)
                local input, label, instance, index = self:getByClass(class)
                while (input == nil) do -- if the sample is nil for some reason (not read properly, size mismatch etc.) sample another one
                    input, label, instance, index = self:getByClass(class) -- !!
                end
                inputs1[i]:copy(input)
                labels[cnt]:copy(label)
                table.insert(instances, instance)
                indices[cnt] = index
                cnt = cnt + 1
             end

            opt.input = 'rgbsegm15joints2Djoints3D'
            --opt.inSize = {(3 + 15 + #opt.jointsIx), opt.sampleRes, opt.sampleRes}
            opt.segm = true
            opt.rgb = true
            opt.joints3D = true
            opt.whiten = false

            for i=1,q2 do
                local class = torch.random(1, #self.classes)
                local input, label, instance, index = self:getByClass(class)
                while (input == nil) do -- if the sample is nil for some reason (not read properly, size mismatch etc.) sample another one
                    input, label, instance, index = self:getByClass(class) -- !!
                end
                inputs2[1][i]:copy(input[1])
                inputs2[2][i]:copy(input[2])
                labels[cnt]:copy(label)
                table.insert(instances, instance)
                indices[cnt] = index
                cnt = cnt + 1
             end

            -- For the validation epoch, put back to rgb input mode
            opt.input = 'rgb'
            opt.inSize = {3, opt.sampleRes, opt.sampleRes}
            opt.rgb = true
            opt.joints3D = false
            opt.whiten = true
        elseif(opt.applyHG == 'joints3D') then
            inputs1 = torch.Tensor(q1, 3, opt.inSize[2], opt.inSize[3]) --
            inputs2 = torch.Tensor(q2, (#opt.jointsIx*opt.depthClasses), opt.heatmapSize, opt.heatmapSize) --
            
            opt.input = 'rgb'
            opt.inSize = {3, opt.sampleRes, opt.sampleRes}
            opt.rgb = true
            opt.whiten = true -- check!

            local cnt = 1
            for i=1,q1 do
                local class = torch.random(1, #self.classes)
                local input, label, instance, index = self:getByClass(class)
                while (input == nil) do -- if the sample is nil for some reason (not read properly, size mismatch etc.) sample another one
                    input, label, instance, index = self:getByClass(class) -- !!
                end
                inputs1[i]:copy(input)
                labels[cnt]:copy(label)
                table.insert(instances, instance)
                indices[cnt] = index
                cnt = cnt + 1
             end

            opt.input = 'joints3D'
            --opt.inSize = {(3 + 15 + #opt.jointsIx), opt.sampleRes, opt.sampleRes}
            opt.rgb = false
            opt.joints3D = true
            opt.whiten = false

            for i=1,q2 do
                local class = torch.random(1, #self.classes)
                local input, label, instance, index = self:getByClass(class)
                while (input == nil) do -- if the sample is nil for some reason (not read properly, size mismatch etc.) sample another one
                    input, label, instance, index = self:getByClass(class) -- !!
                end
                inputs2[i]:copy(input)
                labels[cnt]:copy(label)
                table.insert(instances, instance)
                indices[cnt] = index
                cnt = cnt + 1
             end

            -- For the validation epoch, put back to rgb input mode
            opt.input = 'rgb'
            opt.inSize = {3, opt.sampleRes, opt.sampleRes}
            opt.rgb = true
            opt.joints3D = false
            opt.whiten = true
        elseif(opt.applyHG == 'segm15joints3D') then
            inputs1 = torch.Tensor(q1, 3, opt.inSize[2], opt.inSize[3]) --
            inputs2 = {} -- torch.Tensor(q2, (3+15 + #opt.jointsIx), opt.inSize[2], opt.inSize[3]) --
            table.insert(inputs2, torch.Tensor(q2, (15), opt.inSize[2], opt.inSize[3]))
            table.insert(inputs2, torch.Tensor(q2, opt.depthClasses*#opt.jointsIx, opt.heatmapSize, opt.heatmapSize))

            opt.input = 'rgb'
            opt.inSize = {3, opt.sampleRes, opt.sampleRes}
            opt.rgb = true
            opt.whiten = true -- check!

            local cnt = 1
            for i=1,q1 do
                local class = torch.random(1, #self.classes)
                local input, label, instance, index = self:getByClass(class)
                while (input == nil) do -- if the sample is nil for some reason (not read properly, size mismatch etc.) sample another one
                    input, label, instance, index = self:getByClass(class) -- !!
                end
                inputs1[i]:copy(input)
                labels[cnt]:copy(label)
                table.insert(instances, instance)
                indices[cnt] = index
                cnt = cnt + 1
             end

            opt.input = 'segm15joints3D'
            --opt.inSize = {(3 + 15 + #opt.jointsIx), opt.sampleRes, opt.sampleRes}
            opt.rgb = true
            opt.joints3D = true
            opt.whiten = false

            for i=1,q2 do
                local class = torch.random(1, #self.classes)
                local input, label, instance, index = self:getByClass(class)
                while (input == nil) do -- if the sample is nil for some reason (not read properly, size mismatch etc.) sample another one
                    input, label, instance, index = self:getByClass(class) -- !!
                end
                inputs2[1][i]:copy(input[1])
                inputs2[2][i]:copy(input[2])
                labels[cnt]:copy(label)
                table.insert(instances, instance)
                indices[cnt] = index
                cnt = cnt + 1
             end

            -- For the validation epoch, put back to rgb input mode
            opt.input = 'rgb'
            opt.inSize = {3, opt.sampleRes, opt.sampleRes}
            opt.rgb = true
            opt.joints3D = false
            opt.whiten = true
        end

        inputs = {inputs1, inputs2}
    else -- not mix

        local ninputs = 1
        local noutputs = 1
        local n
        if pcall(function() n = #opt.inSize[1] end) then -- if different types of input
            ninputs = #opt.inSize -- number of inputs
            inputs = {}
            for cnt = 1, ninputs do
                table.insert(inputs, torch.Tensor(quantity, unpack(opt.inSize[cnt]))) --opt.inSize[cnt][1], opt.inSize[cnt][2], opt.inSize[cnt][3]))
            end
        else
            inputs = torch.Tensor(quantity, unpack(opt.inSize)) --opt.inSize[1], opt.inSize[2], opt.inSize[3])
        end

        if pcall(function() n = #opt.outSize[1] end) then -- if different types of output
            noutputs = #opt.outSize -- number of outputs
            labels = {}
            for cnt = 1, noutputs do
                table.insert(labels, torch.Tensor(quantity, unpack(opt.outSize[cnt])))--opt.outSize[cnt][1], opt.outSize[cnt][2], opt.outSize[cnt][3]))
            end
        else
            labels = torch.Tensor(quantity, unpack(opt.outSize))
        end
        
        instances = {}
        indices = torch.Tensor(quantity)
        for i=1,quantity do
            local class = torch.random(1, #self.classes)
            local input, label, instance, index = self:getByClass(class)
            while (input == nil) do -- if the sample is nil for some reason (not read properly, size mismatch etc.) sample another one
                input, label, instance, index = self:getByClass(class) -- !!
            end

            if(ninputs == 1) then
                inputs[i]:copy(input)
            else
                for cnt = 1, ninputs do
                  inputs[cnt][i]:copy(input[cnt])
                end
            end

            if(noutputs == 1) then
                labels[i]:copy(label)
            else
                for cnt = 1, noutputs do
                  labels[cnt][i]:copy(label[cnt])
                end
            end

            table.insert(instances, instance)
            indices[i] = index
        end
    end -- end if mix

    return inputs, labels, instances, indices
end

function dataset:get(i1, i2)
    local indices = torch.range(i1, i2);
    local quantity = i2 - i1 + 1;
    assert(quantity > 0)
    -- now that indices has been initialized, get the samples
    local inputs, labels, instances

    local ninputs = 1
    local noutputs = 1
    local n
    if pcall(function() n = #opt.inSize[1] end) then -- if different types of input
        ninputs = #opt.inSize
        inputs = {}
        for cnt = 1, ninputs do
            table.insert(inputs, torch.Tensor(quantity, unpack(opt.inSize[cnt])))
        end
    else
        inputs = torch.Tensor(quantity, opt.inSize[1], opt.inSize[2], opt.inSize[3])
    end

    if pcall(function() n = #opt.outSize[1] end) then -- if different types of output
        noutputs = #opt.outSize -- number of outputs
        labels = {}
        for cnt = 1, noutputs do
            table.insert(labels, torch.Tensor(quantity, unpack(opt.outSize[cnt])))
        end
    else
        labels = torch.Tensor(quantity, unpack(opt.outSize)):zero()
    end

    instances = {}
    for i=1,quantity do
        local imgpath = ffi.string(torch.data(self.imagePath[indices[i]]))
        local input, label, instance = self:sampleHookTest(imgpath)
        if (input == nil) then -- or label == nil) then
            if(ninputs == 1) then
                input = torch.Tensor(opt.inSize[1], opt.inSize[2], opt.inSize[3]):zero()--+0.5
            else
                input = {}
                for cnt = 1, ninputs do
                    table.insert(input, torch.Tensor(1, unpack(opt.inSize[cnt])):zero())--+0.5)
                end
            end
            if(noutputs == 1) then
                label = torch.Tensor(1, unpack(opt.outSize)):zero()
                if(opt.supervision == 'depth' or opt.supervision == 'segm' or opt.supervision == 'voxels' or opt.supervision == 'partvoxels') then
                    label = label:add(1)
                end
            else
              label = {}
              for cnt = 1, noutputs do
                  table.insert(label, torch.Tensor(1, unpack(opt.outSize[cnt])):zero())
                  if(opt.supervision == 'segm15joints2Djoints3D' or opt.supervision == 'segm15joints2Djoints3Dvoxels'
                    or opt.supervision == 'segm15joints2Djoints3Dpartvoxels') then
                      label[cnt]:add(1) -- segm should not be zero
                  end
              end
            end
            instance = {}
        end
        if(ninputs == 1) then
            inputs[i]:copy(input)
        else
            for cnt = 1, ninputs do
                inputs[cnt][i]:copy(input[cnt])
            end
        end
        if(noutputs == 1) then
            labels[i]:copy(label)
        else
            for cnt = 1, noutputs do
                labels[cnt][i]:copy(label[cnt])
            end
        end
        table.insert(instances, instance)
    end

    return inputs, labels, instances, indices
end

return dataset
