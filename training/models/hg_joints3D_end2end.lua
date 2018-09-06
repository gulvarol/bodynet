paths.dofile('layers/Residual.lua')

local function hourglass(n, f, inp)
    -- Upper branch
    local up1 = inp
    for i = 1,opt.nModules do up1 = Residual(f,f)(up1) end

    -- Lower branch
    local low1 = cudnn.SpatialMaxPooling(2,2,2,2)(inp)
    for i = 1,opt.nModules do low1 = Residual(f,f)(low1) end
    local low2

    if n > 1 then low2 = hourglass(n-1,f,low1)
    else
        low2 = low1
        for i = 1,opt.nModules do low2 = Residual(f,f)(low2) end
    end

    local low3 = low2
    for i = 1,opt.nModules do low3 = Residual(f,f)(low3) end
    local up2 = nn.SpatialUpSamplingNearest(2)(low3)

    -- Bring two branches together
    return nn.CAddTable()({up1,up2})
end

local function lin(numIn,numOut,inp)
    -- Apply 1x1 convolution, stride 1, no padding
    local l = cudnn.SpatialConvolution(numIn,numOut,1,1,1,1,0,0)(inp)
    return cudnn.ReLU(true)(nn.SpatialBatchNormalization(numOut)(l))
end


-- Generic stacked hourglass model
function createSH(nIn, nOut)
    local inp = nn.Identity()()
    -- Initial processing of the image
    local cnv1_ = cudnn.SpatialConvolution(nIn,64,7,7,2,2,3,3)(inp)         -- 128
 
    local cnv1 = cudnn.ReLU(true)(nn.SpatialBatchNormalization(64)(cnv1_))
    local r1 = Residual(64,128)(cnv1)
    local pool = cudnn.SpatialMaxPooling(2,2,2,2)(r1)                       -- 64
    local r4 = Residual(128,128)(pool)
    local r5 = Residual(128,opt.nFeats)(r4)

    local out = {}
    local inter = r5

    for i = 1,opt.nStack do
        local hg = hourglass(4,opt.nFeats,inter)

        -- Residual layers at output resolution
        local ll = hg
        for j = 1,opt.nModules do ll = Residual(opt.nFeats,opt.nFeats)(ll) end
        -- Linear layer to produce first set of predictions
        ll = lin(opt.nFeats,opt.nFeats,ll)

        -- Predicted heatmaps
        local tmpOut = cudnn.SpatialConvolution(opt.nFeats,nOut,1,1,1,1,0,0)(ll)

        -- If    intermediate supervision: add
        -- If no intermediate supervision: add if this is the last stack
        if(  opt.intsup  or  ( (i == opt.nStack) and (not opt.intsup) )  ) then
            table.insert(out,tmpOut)
        end

        -- Add predictions back
        if i < opt.nStack then
            local ll_ = cudnn.SpatialConvolution(opt.nFeats,opt.nFeats,1,1,1,1,0,0)(ll)
            local tmpOut_ = cudnn.SpatialConvolution(nOut,opt.nFeats,1,1,1,1,0,0)(tmpOut)
            inter = nn.CAddTable()({inter, ll_, tmpOut_})
        end
    end

    -- Final model
    local model = nn.gModule({inp}, out)

    print('Return generic stacked hourglass network with ' .. nIn .. ' input, ' .. nOut .. ' output channels.')
    return model

end

-- RGB in -> Segm out
function createSegm(inp)
    return createSH(3, opt.segmClasses)(inp)
end

-- RGB in -> Joints2D out
function createJoints2D(inp)
    return createSH(3, #opt.jointsIx)(inp)
end

-- RGB+Segm+Joints2D in -> Joints3D out
function createJoints3D()
    local inp = nn.Identity()() -- {RGB, Segm + Joints2D}
    local rgb = nn.SelectTable(1)(inp)                                      -- 3 x 256 x 256
    local segm15joints2D = nn.SelectTable(2)(inp)                             -- (15+24) x 64 x 64

    -- Initial processing of the image
    local cnv1_ = cudnn.SpatialConvolution(3,64,7,7,2,2,3,3)(rgb)           -- 64 x 128 x 128
    local cnv1 = cudnn.ReLU(true)(nn.SpatialBatchNormalization(64)(cnv1_))
    local r1 = Residual(64,128)(cnv1)
    local pool = cudnn.SpatialMaxPooling(2,2,2,2)(r1)                       -- 64

    local pool_concat = nn.JoinTable(2)({pool, segm15joints2D})               -- (128 + 15 + 24) x 64 x 64

    local r4 = Residual(128 + opt.segmClasses + #opt.jointsIx,128)(pool_concat)
    local r5 = Residual(128,opt.nFeats)(r4)

    local out = {}
    local inter = r5
    for i = 1,opt.nStack do
        local hg = hourglass(4,opt.nFeats,inter)

        -- Residual layers at output resolution
        local ll = hg
        for j = 1,opt.nModules do ll = Residual(opt.nFeats,opt.nFeats)(ll) end
        -- Linear layer to produce first set of predictions
        ll = lin(opt.nFeats,opt.nFeats,ll)

        -- Predicted heatmaps
        local tmpOut = cudnn.SpatialConvolution(opt.nFeats,opt.depthClasses*#opt.jointsIx,1,1,1,1,0,0)(ll)

        -- If    intermediate supervision: add
        -- If no intermediate supervision: add if this is the last stack
        if(  opt.intsup  or  ( (i == opt.nStack) and (not opt.intsup) )  ) then
            table.insert(out,tmpOut)
        end

        -- Add predictions back
        if i < opt.nStack then
            local ll_ = cudnn.SpatialConvolution(opt.nFeats,opt.nFeats,1,1,1,1,0,0)(ll)
            local tmpOut_ = cudnn.SpatialConvolution(opt.depthClasses*#opt.jointsIx,opt.nFeats,1,1,1,1,0,0)(tmpOut)
            inter = nn.CAddTable()({inter, ll_, tmpOut_})
        end
    end

    -- Final model
    local model = nn.gModule({inp}, out)

    print('Return 3D pose network')
    return model
end

function createModel()
    print('Loading pre-trained segmentation network: ' .. opt.modelSegm)
    local modelSegm     = torch.load(opt.modelSegm)
    print('Loading pre-trained joints2D network: ' .. opt.modelJoints2D)
    local modelJoints2D = torch.load(opt.modelJoints2D)
    print('Loading pre-trained joints3D network: ' .. opt.modelJoints3D)
    local modelJoints3D = torch.load(opt.modelJoints3D)

    local rgb = nn.Identity()()
    --local segmout = createSH(3, opt.segmClasses)(rgb) -- createSegm()(rgb) -- 8 x 15 x 64 x 64
    local segmOut = (modelSegm)(rgb)
    --local joints2Dout = createSH(3, #opt.jointsIx)(rgb)  -- createJoints2D()(rgb) -- 8 x 24 x 64 x 64
    local joints2DOut = (modelJoints2D)(rgb)

    local segmInp = nn.SelectTable(opt.nStackSegm)(segmOut) -- take the last stack's output 15 x 64 x 64
    local joints2DInp = nn.SelectTable(opt.nStackJoints2D)(joints2DOut) -- take the last stack's output 24 x 64 x 64

    local segmInpUp = nn.SpatialUpSamplingBilinear({oheight=opt.sampleRes, owidth=opt.sampleRes})(segmInp) -- upsample to make 15 x 256 x 256
    local joints2DInpUp = nn.SpatialUpSamplingBilinear({oheight=opt.sampleRes, owidth=opt.sampleRes})(joints2DInp) -- upsample to make 24 x 256 x 256

    local segmInpNorm = cudnn.SpatialSoftMax():cuda()(segmInpUp) 
    
    local rgbsegm15joints2DInp = nn.JoinTable(2)({rgb, segmInpNorm, joints2DInpUp}) -- (3 + 15 + 24) x 256 x 256

    local joints3DOut = (modelJoints3D)(rgbsegm15joints2DInp)
    --local joints3Dout = createJoints3D()(rgbsegm15joints2DInp) -- 8 x 65 * 24 x 64 x 64

    local out = {}
    for st = 1, opt.nStack do
        table.insert(out, nn.SelectTable(st)(segmOut))
        table.insert(out, nn.SelectTable(st)(joints2DOut))
        table.insert(out, nn.SelectTable(st)(joints3DOut))
    end

    local model = nn.gModule({rgb}, out)
    print('Return end-to-end rgb->joints2D, segm->joints3D network')
    return model
end

