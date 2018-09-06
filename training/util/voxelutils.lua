paths.dofile('img.lua')

function alignGTVoxels(voxels, silh, joints2D, path)
    local volsize = voxels:size() -- 44 69 128 or 43 128 43 ...

    local tBox = getTightBoxSegm(silh) -- segm will be mostly 256*256

    local s =  opt.nVoxels / silh:size(2)
    -- 128/256 * segmGap / volGap 
    local scalingx = s * (tBox.x_max - tBox.x_min) / volsize[2] 
    local scalingy = s * (tBox.y_max - tBox.y_min) / volsize[1] 
    local scaling
    if(volsize[2] <= volsize[1]) then
       scaling = scalingy
       --if volsize[1] ~= 128 then print(volsize); print(path) end
    elseif(volsize[2] > volsize[1]) then
       scaling = scalingx
       --if volsize[2] ~= 128 then print(volsize); print(path) end
    end

    --if opt.show then showVoxels(voxels) end
    local voxelsScaledXY = image.scale(voxels:permute(3, 2, 1), torch.round(scaling*volsize[1]), torch.round(scaling*volsize[2]), 'simple'):permute(3, 2, 1) -- width, height
    --if opt.show then showVoxels(voxelsScaledXY) end
    local voxelsScaledZ = image.scale(voxelsScaledXY, torch.round(scaling*volsize[3]), voxelsScaledXY:size(2), 'simple')

    local tBoxScaled = {}
    tBoxScaled.x_min = torch.round(tBox.x_min * s)
    tBoxScaled.x_max = torch.round(tBox.x_max * s)
    tBoxScaled.y_min = torch.round(tBox.y_min * s)
    tBoxScaled.y_max = torch.round(tBox.y_max * s)

    local x1, x2, y1, y2, z1, z2
    -- X padding
    if(joints2D[{{}, {2}}]:lt(0):any()) then -- any negative y values in 2D joints?
        if(opt.verbose) then print('negative y') end
        x1 = tBoxScaled.y_max - voxelsScaledXY:size(1) + 1
        x2 = tBoxScaled.y_max
    else
        x1 = tBoxScaled.y_min
        x2 = tBoxScaled.y_min + voxelsScaledXY:size(1) - 1
    end
    -- Y padding
    if(joints2D[{{}, {1}}]:lt(0):any()) then -- any negative x values in 2D joints?
        if(opt.verbose) then print('negative x') end
        y1 = tBoxScaled.x_max - voxelsScaledXY:size(2) + 1
        y2 = tBoxScaled.x_max
    else
        y1 = tBoxScaled.x_min
        y2 = tBoxScaled.x_min + voxelsScaledXY:size(2) - 1
    end
    -- Z padding
    if(voxelsScaledZ:size(3) > opt.nVoxels) then -- trim
        local ztrim = torch.round((voxelsScaledZ:size(3) - opt.nVoxels)/2) + 1
        voxelsScaledZ = voxelsScaledZ[{{}, {}, {ztrim, ztrim + opt.nVoxels -1}}]
        z1 = 1
        z2 = voxelsScaledZ:size(3)
        if(opt.verbose) then print('Z too big ' .. path) end
    else -- pad
        z1 = torch.round((opt.nVoxels - voxelsScaledZ:size(3))/2) + 1
        z2 = z1 + voxelsScaledZ:size(3) -1
    end

    local voxelsPadded = torch.zeros(opt.nVoxels, opt.nVoxels, opt.nVoxels):byte()

    if pcall(function() 
          voxelsPadded[{{x1, x2}, {y1, y2}, {z1, z2}}] = voxelsScaledZ -- change this
        end) then
    else
        if(opt.verbose) then
            -- PRINT STUFF
            print(volsize)
            print(voxelsScaledXY:size())
            print(voxelsScaledZ:size())
            print(s, scalingx, scalingy, scaling)
            print(tBox.x_min, tBox.x_max, tBox.y_min, tBox.y_max)
            print(tBoxScaled.x_min, tBoxScaled.x_max, tBoxScaled.y_min, tBoxScaled.y_max)
            print(x1, x2, y1, y2, z1, z2)
        end
    end

    local trans_params = {}
    trans_params.x1 = x1
    trans_params.x2 = x2
    trans_params.y1 = y1
    trans_params.y2 = y2
    trans_params.z1 = z1
    trans_params.z2 = z2
    trans_params.scaling = scaling

    return voxelsPadded, trans_params
end