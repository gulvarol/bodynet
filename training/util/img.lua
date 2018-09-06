function applyFn(fn, t, t2)
    -- Apply an operation whether passed a table or tensor
    local t_ = {}
    if type(t) == "table" then
        if t2 then
            for i = 1,#t do t_[i] = applyFn(fn, t[i], t2[i]) end
        else
            for i = 1,#t do t_[i] = applyFn(fn, t[i]) end
        end
    else t_ = fn(t, t2) end
    return t_
end

-------------------------------------------------------------------------------
-- Coordinate transformation
-------------------------------------------------------------------------------

function getTransform(center, scale, rot, res)
    local h = 200 * scale
    local t = torch.eye(3)

    -- Scaling
    t[1][1] = res / h
    t[2][2] = res / h

    -- Translation
    t[1][3] = res * (-center[1] / h + .5)
    t[2][3] = res * (-center[2] / h + .5)

    -- Rotation
    if rot ~= 0 then
        rot = -rot
        local r = torch.eye(3)
        local ang = rot * math.pi / 180
        local s = math.sin(ang)
        local c = math.cos(ang)
        r[1][1] = c
        r[1][2] = -s
        r[2][1] = s
        r[2][2] = c
        -- Need to make sure rotation is around center
        local t_ = torch.eye(3)
        t_[1][3] = -res/2
        t_[2][3] = -res/2
        local t_inv = torch.eye(3)
        t_inv[1][3] = res/2
        t_inv[2][3] = res/2
        t = t_inv * r * t_ * t
    end

    return t
end

function transform(pt, center, scale, rot, res, invert)
    local pt_ = torch.ones(3)
    pt_[1],pt_[2] = pt[1]-1,pt[2]-1

    local t = getTransform(center, scale, rot, res)
    if invert then
        t = torch.inverse(t)
    end
    local new_point = (t*pt_):sub(1,2):add(1e-4)

    return new_point:int():add(1)
end

function transformPreds(coords, center, scale, res)
    local origDims = coords:size()
    coords = coords:view(-1,2)
    local newCoords = coords:clone()
    for i = 1,coords:size(1) do
        newCoords[i] = transform(coords[i], center, scale, 0, res, 1)
    end
    return newCoords:view(origDims)
end

-------------------------------------------------------------------------------
-- Cropping
-------------------------------------------------------------------------------

function checkDims(dims)
    return dims[3] < dims[4] and dims[5] < dims[6]
end

function crop(img, center, scale, rot, res, method)
    if(type(center) == 'table') then center = torch.Tensor(center) end
    local ndim = img:nDimension()
    if ndim == 2 then img = img:view(1,img:size(1),img:size(2)) end -- if grayscale
    local ht,wd = img:size(2), img:size(3) -- 240 , 320
    local tmpImg,newImg = img, torch.zeros(img:size(1), res, res)

    -- Modify crop approach depending on whether we zoom in/out
    -- This is for efficiency in extreme scaling cases
    local scaleFactor = (200 * scale) / res -- why 200?
    if scaleFactor < 2 then scaleFactor = 1
    else
        local newSize = math.floor(math.max(ht,wd) / scaleFactor)
        if newSize < 2 then
           -- Zoomed out so much that the image is now a single pixel or less
           if ndim == 2 then newImg = newImg:view(newImg:size(2),newImg:size(3)) end
           return newImg
        else
            tmpImg = image.scale(img,newSize, method)
            ht,wd = tmpImg:size(2),tmpImg:size(3)
        end
    end

    -- Calculate upper left and bottom right coordinates defining crop region
    --local c,s = center:float()/scaleFactor, scale/scaleFactor
    local c,s = center/scaleFactor, scale/scaleFactor
    local ul = transform({1,1}, c, s, 0, res, true)
    local br = transform({res+1,res+1}, c, s, 0, res, true)
    if scaleFactor >= 2 then br:add(-(br - ul - res)) end

    -- If the image is to be rotated, pad the cropped area
    local pad = math.ceil(torch.norm((ul - br):float())/2 - (br[1]-ul[1])/2)
    if rot ~= 0 then ul:add(-pad); br:add(pad) end

    -- Define the range of pixels to take from the old image
    local old_ = {1,-1,math.max(1, ul[2]), math.min(br[2], ht+1) - 1,
                       math.max(1, ul[1]), math.min(br[1], wd+1) - 1}
    -- And where to put them in the new image
    local new_ = {1,-1,math.max(1, -ul[2] + 2), math.min(br[2], ht+1) - ul[2],
                       math.max(1, -ul[1] + 2), math.min(br[1], wd+1) - ul[1]}

    -- Initialize new image and copy pixels over
    local newImg = torch.zeros(img:size(1), br[2] - ul[2], br[1] - ul[1])
    if not pcall(function() newImg:sub(unpack(new_)):copy(tmpImg:sub(unpack(old_))) end) then
       print("Error occurred during crop!")
    end
    if rot ~= 0 then
        -- Rotate the image and remove padded area
        newImg = image.rotate(newImg, rot*math.pi / 180, method)
        newImg = newImg:sub(1,-1,pad+1,newImg:size(2)-pad,pad+1,newImg:size(3)-pad):clone()
    end

    if scaleFactor < 2 then newImg = image.scale(newImg,res,res, method) end
    
    if ndim == 2 then newImg = newImg:view(newImg:size(2),newImg:size(3)) end

    return newImg
end

function twoPointCrop(img, s, pt1, pt2, pad, res)
    local center = (pt1 + pt2) / 2
    local scale = math.max(20*s,torch.norm(pt1 - pt2)) * .007
    scale = scale * pad
    local angle = math.atan2(pt2[2]-pt1[2],pt2[1]-pt1[1]) * 180 / math.pi - 90
    return crop(img, center, scale, angle, res)
end


-------------------------------------------------------------------------------
-- Draw 3D gaussian
-------------------------------------------------------------------------------

function meshgrid3(x,y,z)
   local xx = torch.repeatTensor(x, y:size(1), z:size(1),1)
   local yy = torch.repeatTensor(y:view(-1, 1), x:size(1), 1, z:size(1))
   local zz = torch.repeatTensor(z:view(-1, 1, 1), 1, x:size(1), y:size(1)):view(x:size(1), y:size(1), z:size(1))
    return xx, yy, zz
end

-- Creates a 3D Gaussian kernel similar to image.gauss function. exp( - (x^2+y^2+z^2) / (2*sigma^2) )
function gaussian3D(k)
   siz = (k-1)/2
   sigma = 0.25 -- default from image.gauss
   xx = torch.linspace(-siz,siz,k)
   yy = torch.linspace(-siz,siz,k)
   zz = torch.linspace(-siz,siz,k)
   X, Y, Z = meshgrid3(xx, yy, zz)

   sig = sigma*k -- the definition of sigma in image.gauss is different from e.g. Matlab
   h = -0.5*(torch.cmul(X, X) + torch.cmul(Y, Y) + torch.cmul(Z, Z))/(sig*sig) 
   h = torch.exp(h)
   h[h:le((1e-16)*h:max())] = 0;
   
   --sumh = h:sum()
   --if sumh ~= 0 then
   --   h  = h/sumh -- sums to 1
   --end
   h = h/h:max() -- max is 1
   return h
end

-- Places a 3D Gaussian kernel at the specified point (pt) inside 3D volume (img) with given variance (sigma)
-- Example usage: output = draw3DGaussian(torch.zeros(20, 64, 64), {12, 60, 45}, 1)
-- img: z, y, x
-- pt: z, y, x
function draw3DGaussian(img, pt, sigma)
    -- Draw a 3D gaussian
    -- Check that any part of the gaussian is in-bounds
    local ul = {math.floor(pt[1] - 3 * sigma), math.floor(pt[2] - 3 * sigma), math.floor(pt[3] - 3 * sigma)}
    local br = {math.floor(pt[1] + 3 * sigma), math.floor(pt[2] + 3 * sigma), math.floor(pt[3] + 3 * sigma)}
    -- If not, return the image as is
   if (ul[1] > img:size(1)
       or ul[2] > img:size(2)
       or ul[3] > img:size(3) 
       or br[1] < 1
       or br[2] < 1
       or br[3] < 1) then return img end
    -- Generate gaussian
    local size = 6 * sigma + 1
   local g = gaussian3D(size)
    -- Usable gaussian range
    local g_x = {math.max(1, -ul[1]), math.min(br[1], img:size(1)) - math.max(1, ul[1]) + math.max(1, -ul[1])}
    local g_y = {math.max(1, -ul[2]), math.min(br[2], img:size(2)) - math.max(1, ul[2]) + math.max(1, -ul[2])}
    local g_z = {math.max(1, -ul[3]), math.min(br[3], img:size(3)) - math.max(1, ul[3]) + math.max(1, -ul[3])}
    -- Image range
    local img_x = {math.max(1, ul[1]), math.min(br[1], img:size(1))}
    local img_y = {math.max(1, ul[2]), math.min(br[2], img:size(2))}
    local img_z = {math.max(1, ul[3]), math.min(br[3], img:size(3))}
    assert(g_x[1] > 0 and g_y[1] > 0 and g_z[1] > 0)
    img:sub(img_x[1], img_x[2], img_y[1], img_y[2], img_z[1], img_z[2]):add(g:sub(g_x[1], g_x[2], g_y[1], g_y[2], g_z[1], g_z[2]))
    img[img:gt(1)] = 1
    return img
end

-------------------------------------------------------------------------------
-- Draw 2D gaussian
-------------------------------------------------------------------------------

function drawGaussian(img, pt, sigma)
    -- Draw a 2D gaussian
    -- Check that any part of the gaussian is in-bounds
    local ul = {math.floor(pt[1] - 3 * sigma), math.floor(pt[2] - 3 * sigma)}
    local br = {math.floor(pt[1] + 3 * sigma), math.floor(pt[2] + 3 * sigma)}
    -- If not, return the image as is
    if (ul[1] > img:size(2) or ul[2] > img:size(1) or br[1] < 1 or br[2] < 1) then return img end
    -- Generate gaussian
    local size = 6 * sigma + 1
    local g = image.gaussian(size) -- , 1 / size, 1)
    -- Usable gaussian range
    local g_x = {math.max(1, -ul[1]), math.min(br[1], img:size(2)) - math.max(1, ul[1]) + math.max(1, -ul[1])}
    local g_y = {math.max(1, -ul[2]), math.min(br[2], img:size(1)) - math.max(1, ul[2]) + math.max(1, -ul[2])}
    -- Image range
    local img_x = {math.max(1, ul[1]), math.min(br[1], img:size(2))}
    local img_y = {math.max(1, ul[2]), math.min(br[2], img:size(1))}
    assert(g_x[1] > 0 and g_y[1] > 0)
    img:sub(img_y[1], img_y[2], img_x[1], img_x[2]):add(g:sub(g_y[1], g_y[2], g_x[1], g_x[2]))
    img[img:gt(1)] = 1
    return img
end

-------------------------------------------------------------------------------
-- Flipping functions
-------------------------------------------------------------------------------

function shuffleLRJoints(x)
    local dim = 1
    assert(x:nDimension() == 2 or x:nDimension() == 3 or x:nDimension() == 4)
    if x:nDimension() == 4 then
    --    print('Flip 3D')
    --    dim = 2
    elseif x:nDimension() == 3 then
    --    assert(x:nDimension() == 3)
    --    print('Flip 2D')
    --    dim = 1
    else
          assert(x:nDimension() == 2)
    --    print('Joint coords')
    end
    local matched_parts 
    if(#opt.jointsIx == 16) then
        matched_parts = {
            {1,6},   {2,5},   {3,4},
            {11,16}, {12,15}, {13,14}
       }
    elseif(#opt.jointsIx == 24) then
        matched_parts = {
            {2,3}, {5,6}, {8,9},
            {11,12}, {14,15}, {17,18},
            {19, 20}, {21, 22}, {23, 24}
        }
    end

    for i = 1,#matched_parts do
        local idx1, idx2 = unpack(matched_parts[i])
        local tmp = x:narrow(dim, idx1, 1):clone()
        x:narrow(dim, idx1, 1):copy(x:narrow(dim, idx2, 1))
        x:narrow(dim, idx2, 1):copy(tmp)
    end
    return x
end

function shuffleLRSegm(x)
    local dim
    if x:nDimension() == 2 then
        local s = {1, 2, 3, 7, 8, 9, 4, 5, 6, 13, 14, 15, 10, 11, 12}
        local out = torch.zeros(x:size())
        for i = 1,#s do out[x:eq(i)] = s[i] end
        return out
    else
        assert(x:nDimension() == 3)
        dim = 1
    end
    local matched_parts
    if(opt.segmClasses == 15) then
        matched_parts= {
        --{9,12},   {10,13},   {11,14},
        --{3,6}, {4,7}, {5,8}
        {10,13},   {11,14},   {12,15},
        {4,7}, {5,8}, {6,9}
        }
    end
    for i = 1,#matched_parts do
        local idx1, idx2 = unpack(matched_parts[i])
        local tmp = x:narrow(dim, idx1, 1):clone()
        x:narrow(dim, idx1, 1):copy(x:narrow(dim, idx2, 1))
        x:narrow(dim, idx2, 1):copy(tmp)
    end
    return x
end

function flip(x)
    require 'image'
    local y = torch.FloatTensor(x:size())
    for i = 1, x:size(1) do
        image.hflip(y[i], x[i]:float())
    end
    return y:typeAs(x)
end

function segm15Img(segm)
    local img = torch.zeros(segm:size(2), segm:size(3))
    for i = 1,segm:size(1) do -- for each body part
        img[segm[i]:eq(1)] = i
    end
    return img
end


function getTightBoxSegm(silh)
    -- silh: a 0/1 bg/fg segmentation mask
    -- Tighest bounding box covering the segmentation map
    local tBox = {}
    -- Sum over y --> range in x
    local rx = silh:sum(1)
    -- Find indices of nonzero elements
    local a = rx:nonzero()
    -- The first nonzero element is the min
    tBox.x_min = a[1][2]
    -- The last nonzero element is the max
    tBox.x_max = a[-1][2]

    -- Sum over x --> range in y
    local ry = silh:sum(2)
    -- Find indices of nonzero elements
    local b = ry:nonzero()
    -- The first nonzero element is the min
    tBox.y_min = b[1][1]
    -- The last nonzero element is the max
    tBox.y_max = b[-1][1]

    -- Human width and height
    tBox.humWidth  = tBox.x_max - tBox.x_min + 1
    tBox.humHeight = tBox.y_max - tBox.y_min + 1

    return tBox
end

function getTightBox(label)
    -- Tighest bounding box covering the joint positions
    local tBox = {}
    tBox.x_min = label[{{}, {1}}]:min()
    tBox.y_min = label[{{}, {2}}]:min()
    tBox.x_max = label[{{}, {1}}]:max()
    tBox.y_max = label[{{}, {2}}]:max()
    tBox.humWidth  = tBox.x_max - tBox.x_min + 1
    tBox.humHeight = tBox.y_max - tBox.y_min + 1

    -- Slightly larger area to cover the head/feet of the human
    tBox.x_min = tBox.x_min - 0.25*tBox.humWidth -- left
    tBox.y_min = tBox.y_min - 0.35*tBox.humHeight -- top
    tBox.x_max = tBox.x_max + 0.25*tBox.humWidth -- right
    tBox.y_max = tBox.y_max + 0.25*tBox.humHeight -- bottom
    tBox.humWidth  = tBox.x_max - tBox.x_min + 1
    tBox.humHeight = tBox.y_max - tBox.y_min +1

    return tBox
end

function getCenter(label)
    local tBox = getTightBox(label)
    local center_x = tBox.x_min + tBox.humWidth/2
    local center_y = tBox.y_min + tBox.humHeight/2

    return {center_x, center_y}
end

--imsize = [3 imHeight imWidth] -- /imWidth is not used on purpose
function getScale(label, imHeight)
    local tBox = getTightBox(label)
    return math.max(tBox.humHeight/240, tBox.humWidth/240)
    --return math.max(tBox.humHeight/imHeight, tBox.humWidth/imHeight)/256 
end