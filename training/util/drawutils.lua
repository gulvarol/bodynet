require 'image'
require 'gnuplot'

function draw3DPose(joints3D, win)

    local pairRef = {
        {1,2},      {2,3},      {3,7},
        {4,5},      {4,7},      {5,6},
        {7,9},      {9,10},
        {14,9},     {11,12},    {12,13},
        {13,9},     {14,15},    {15,16}
    }

    local nlimbs = #pairRef --limbs:size(1)
    win = win or 1
    gnuplot.figure(win)
    local xc = 1 -- 1
    local yc = 2 -- 3
    local zc = 3 -- 2
--    joints3D[{{}, {zc}}] = - joints3D[{{}, {zc}}]
--    joints3D[{{}, {yc}}] = - joints3D[{{}, {yc}}]
        
    local xyz = {}
    for ll = 1, nlimbs do
        table.insert(xyz, { joints3D[{{}, {xc}}]:squeeze():index(1, torch.LongTensor({pairRef[ll][1], pairRef[ll][2]})),
                            joints3D[{{}, {yc}}]:squeeze():index(1, torch.LongTensor({pairRef[ll][1], pairRef[ll][2]})),
                            joints3D[{{}, {zc}}]:squeeze():index(1, torch.LongTensor({pairRef[ll][1], pairRef[ll][2]}))})
        gnuplot.raw('set style line '.. ll ..' lt rgb "cyan"') --  lw 3     pt 6 ps 0.5 -lt rgb "cyan"
    end
    gnuplot.scatter3(xyz)

    gnuplot.xlabel('x') --('z')
    gnuplot.ylabel('y') --('x')
    gnuplot.zlabel('z') --('y')
    gnuplot.raw('set xrange ['.. joints3D[{{}, {xc}}]:min() - 0.5 ..':'..joints3D[{{}, {xc}}]:max() + 0.5 ..']') -- reverse
    gnuplot.raw('set yrange ['.. joints3D[{{}, {yc}}]:min() - 0.5 ..':'..joints3D[{{}, {yc}}]:max() + 0.5 ..']')
    gnuplot.raw('set zrange ['.. joints3D[{{}, {zc}}]:min() - 0.5 ..':'..joints3D[{{}, {zc}}]:max() + 0.5 ..']')

    gnuplot.raw('set view equal xyz')
    gnuplot.raw('set view 15, 160')
end

function showJoints3DHeatmap(joints3Dhm)
    local sumheatmap = joints3Dhm:view(16, 19, 64, 64):sum(1):squeeze()
    local voximgXY = sumheatmap:max(3):squeeze()
    local voximgXZ = sumheatmap:max(2):squeeze()
    local voximgYZ = sumheatmap:max(1):squeeze()
    voximgXY = voximgXY:repeatTensor(3, 1, 1)
    voximgXZ = voximgXZ:repeatTensor(3, 1, 1)
    voximgYZ = voximgYZ:repeatTensor(3, 1, 1)
    local voximgXYf = image.drawRect(voximgXY, 1, 1, voximgXY:size(3), voximgXY:size(2))
    local voximgXZf = image.drawRect(voximgXZ, 1, 1, voximgXZ:size(3), voximgXZ:size(2))
    local voximgYZf = image.drawRect(voximgYZ, 1, 1, voximgYZ:size(3), voximgYZ:size(2))    
    return torch.cat({voximgXYf, voximgXZf, voximgYZf}, 2)
end

function draw2DPoseFrom3DJoints(coords)
    coords = (150*coords:add(-coords:min())):round()+5
    --coords[{{}, {1}}] = 300 - coords[{{}, {1}}]
    return drawSkeleton(torch.ones(3, 300, 300), dummy, coords)
end

function drawLine(img,pt1,pt2,width,color)
    -- I'm sure there's a line drawing function somewhere in Torch,
    -- but since I couldn't find it here's my basic implementation
    local color = color or {1,1,1}
    local m = torch.dist(pt1,pt2)
    local dy = (pt2[2] - pt1[2])/m
    local dx = (pt2[1] - pt1[1])/m
    for j = 1,width do
        local start_pt1 = torch.Tensor({pt1[1] + (-width/2 + j-1)*dy, pt1[2] - (-width/2 + j-1)*dx})
        start_pt1:ceil()
        for i = 1,torch.ceil(m) do
            local y_idx = torch.ceil(start_pt1[2]+dy*i)
            local x_idx = torch.ceil(start_pt1[1]+dx*i)
            if y_idx - 1 > 0 and x_idx -1 > 0 and y_idx < img:size(2) and x_idx < img:size(3) then
                img:sub(1,1,y_idx-1,y_idx,x_idx-1,x_idx):fill(color[1])
                img:sub(2,2,y_idx-1,y_idx,x_idx-1,x_idx):fill(color[2])
                img:sub(3,3,y_idx-1,y_idx,x_idx-1,x_idx):fill(color[3])
            end
        end 
    end
    img[img:gt(1)] = 1

    return img
end

function heatmapVisualization(set, idx, pred, inp, gt)
    local set = set or 'valid'
    local hmImg
    local tmpInp,tmpHm
    if not inp then
        inp, gt = loadData(set,{idx})
        inp = inp[1]
        gt = gt[1][1]
        tmpInp,tmpHm = inp,gt
    else
        tmpInp = inp
        tmpHm = gt or pred
    end
    local nOut,res = tmpHm:size(1),tmpHm:size(3)
    -- Repeat input image, and darken it to overlay heatmaps
    tmpInp = image.scale(tmpInp,res):mul(.3)
    tmpInp[1][1][1] = 1
    hmImg = tmpInp:repeatTensor(nOut,1,1,1)
    if gt then -- Copy ground truth heatmaps to red channel
        hmImg:sub(1,-1,1,1):add(gt:clone():mul(.7))
    end
    if pred then -- Copy predicted heatmaps to blue channel
        hmImg:sub(1,-1,3,3):add(pred:clone():mul(.7))
    end
    -- Rescale so it is a little easier to see
    hmImg = image.scale(hmImg:view(nOut*3,res,res),256):view(nOut,3,256,256)
    return hmImg, inp
end

function drawSkeleton(input, hms, coords)
    local im = input:clone()

    local pairRef = {
        {1,2},      {2,3},      {3,7},
        {4,5},      {4,7},      {5,6},
        {7,9},      {9,10},
        {14,9},     {11,12},    {12,13},
        {13,9},     {14,15},    {15,16}
    }

    local partNames = {'RAnk','RKne','RHip','LHip','LKne','LAnk',
                       'Pelv','Thrx','Neck','Head',
                       'RWri','RElb','RSho','LSho','LElb','LWri'}
    local partColor = {1,1,1,2,2,2,0,0,0,0,3,3,3,4,4,4}

    local actThresh = 0.002 ---5 --0--0.0002 -- was 0.002

    -- Loop through adjacent joint pairings
    for i = 1,#pairRef do
        --if hms[pairRef[i][1]]:mean() > actThresh and hms[pairRef[i][2]]:mean() > actThresh then
            -- Set appropriate line color
            local color
            if partColor[pairRef[i][1]] == 1 then color = {0,.3,1}
            elseif partColor[pairRef[i][1]] == 2 then color = {1,.3,0}
            elseif partColor[pairRef[i][1]] == 3 then color = {0,0,1}
            elseif partColor[pairRef[i][1]] == 4 then color = {1,0,0}
            else color = {.7,0,.7} end
            -- Draw line
            im = drawLine(im, coords[pairRef[i][1]], coords[pairRef[i][2]], 4, color, 0)
        --end
    end
    return im
end

function compileImages(imgs, nrows, ncols, res)
    -- Assumes the input images are all square/the same resolution
    local totalImg = torch.zeros(3,nrows*res,ncols*res)
    for i = 1,#imgs do
        local r = torch.floor((i-1)/ncols) + 1
        local c = ((i - 1) % ncols) + 1
        totalImg:sub(1,3,(r-1)*res+1,r*res,(c-1)*res+1,c*res):copy(imgs[i])
    end
    return totalImg
end

function colorHM(x)
    -- Converts a one-channel grayscale image to a color heatmap image
    local function gauss(x,a,b,c)
        return torch.exp(-torch.pow(torch.add(x,-b),2):div(2*c*c)):mul(a)
    end
    local cl = torch.zeros(3,x:size(1),x:size(2))
    cl[1] = gauss(x,.5,.6,.2) + gauss(x,1,.8,.3)
    cl[2] = gauss(x,1,.5,.3)
    cl[3] = gauss(x,1,.2,.3)
    cl[cl:gt(1)] = 1
    return cl
end

function drawOutput(input, hms, coords)
    local im = drawSkeleton(input, hms, coords)

    local colorHms = {}
    local inp64 = image.scale(input,64):mul(.3)
    for i = 1,16 do 
        colorHms[i] = colorHM(hms[i])
        --colorHms[i] = image.y2jet((hms[i]:add(-hms[i]:mean()):mul(1/(hms[i]:max() - hms[i]:min()))+1)*255)
        colorHms[i]:mul(.7):add(inp64)
    end
    local totalHm = compileImages(colorHms, 4, 4, 64)
    im = compileImages({im,totalHm}, 1, 2, 256)
    im = image.scale(im,756)
    return im
end

function displayPCK(dists, part_idx, label, title, show_key)
    -- Generate standard PCK plot
    if not (type(part_idx) == 'table') then
        part_idx = {part_idx}
    end

    curve_res = 11
    num_curves = #dists
    local t = torch.linspace(0,.5,curve_res)
    local pdj_scores = torch.zeros(num_curves, curve_res)
    local plot_args = {}
    print(title)
    for curve = 1,num_curves do
        for i = 1,curve_res do
            t[i] = (i-1)*.05
            local acc = 0.0
            for j = 1,#part_idx do
                acc = acc + distAccuracy(dists[curve][part_idx[j]], t[i])
            end
            pdj_scores[curve][i] = acc / #part_idx
        end
        plot_args[curve] = {label[curve],t,pdj_scores[curve],'-'}
        print(label[curve],pdj_scores[curve][curve_res])
    end

    gnuplot.raw('set title "' .. title .. '"')
    if not show_key then gnuplot.raw('unset key') 
    else gnuplot.raw('set key font ",6" right bottom') end
    gnuplot.raw('set xrange [0:.5]')
    gnuplot.raw('set yrange [0:1]')
    gnuplot.plot(unpack(plot_args))
end

function showPartVoxels(voxels)
    local voximgXY = voxels:max(3):squeeze()
    local voximgXZ = voxels:max(2):squeeze()
    local voximgYZ = voxels:max(1):squeeze()
    --voximgXY = voximgXY:repeatTensor(3, 1, 1)
    --voximgXZ = voximgXZ:repeatTensor(3, 1, 1)
    --voximgYZ = voximgYZ:repeatTensor(3, 1, 1)
    --local voximgXYf = image.drawRect(voximgXY, 1, 1, voximgXY:size(3), voximgXY:size(2))
    --local voximgXZf = image.drawRect(voximgXZ, 1, 1, voximgXZ:size(3), voximgXZ:size(2))
    --local voximgYZf = image.drawRect(voximgYZ, 1, 1, voximgYZ:size(3), voximgYZ:size(2))
    return {image.y2jet(voximgXY), image.y2jet(voximgXZ), image.y2jet(voximgYZ)}
end


function showVoxels(voxels, bg)
    bg = bg or false
    local volsize = voxels:size()
    local voximgXY, voximgXZ, voximgYZ
    if(bg) then
        voximgXY = voxels:min(3):squeeze()
        voximgXZ = voxels:min(2):squeeze()
        voximgYZ = voxels:min(1):squeeze()
    else
        voximgXY = voxels:max(3):squeeze()
        voximgXZ = voxels:max(2):squeeze()
        voximgYZ = voxels:max(1):squeeze()
    end
    voximgXY = voximgXY:repeatTensor(3, 1, 1)
    voximgXZ = voximgXZ:repeatTensor(3, 1, 1)
    voximgYZ = voximgYZ:repeatTensor(3, 1, 1)
    local voximgXYf = image.drawRect(voximgXY, 1, 1, voximgXY:size(3), voximgXY:size(2))
    local voximgXZf = image.drawRect(voximgXZ, 1, 1, voximgXZ:size(3), voximgXZ:size(2))
    local voximgYZf = image.drawRect(voximgYZ, 1, 1, voximgYZ:size(3), voximgYZ:size(2))
    return {voximgXYf, voximgXZf, voximgYZf}
end