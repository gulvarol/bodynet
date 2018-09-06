paths.dofile('layers/Residual.lua')

local function lin(numIn,numOut,inp)
    -- Apply 1x1 convolution, stride 1, no padding
    local l = cudnn.SpatialConvolution(numIn,numOut,1,1,1,1,0,0)(inp)
    return cudnn.ReLU(true)(nn.SpatialBatchNormalization(numOut)(l))
end

local function linVector(numIn,numOut,inp)
    local l = nn.Linear(numIn,numOut):cuda()(inp)
    return cudnn.ReLU(true)(cudnn.BatchNormalization(numOut)(l))
end

function createModel()
    local inp = nn.Identity()()

    -- Initial processing of the image
    local cnv1_
    if(opt.applyHG == 'segm15') then
        cnv1_ = cudnn.SpatialConvolution(15,64,7,7,2,2,3,3)(inp)                 -- 128
    elseif(opt.applyHG == 'joints2D') then
        cnv1_ = cudnn.SpatialConvolution(#opt.jointsIx,64,7,7,2,2,3,3)(inp)      -- 128
    elseif(opt.applyHG == 'segm15joints2D') then
        cnv1_ = cudnn.SpatialConvolution(15+#opt.jointsIx,64,7,7,2,2,3,3)(inp)   -- 128
    elseif(opt.applyHG == 'rgbsegm15joints2D') then
        cnv1_ = cudnn.SpatialConvolution(3+15+#opt.jointsIx,64,7,7,2,2,3,3)(inp) -- 128
    else
        cnv1_ = cudnn.SpatialConvolution(opt.inSize[1],64,7,7,2,2,3,3)(inp)      -- 128
    end

    -- Initial processing of the image
    local cnv1 = cudnn.ReLU(true)(nn.SpatialBatchNormalization(64)(cnv1_))
    local r1 = Residual(64,128)(cnv1)
    local pool = cudnn.SpatialMaxPooling(2,2,2,2)(r1)  -- 64
    local r2 = Residual(128, 128)(pool)

    local cnv2_ = cudnn.SpatialConvolution(128,128,3,3,1,1,1,1)(r2)
    local cnv2 = cudnn.ReLU(true)(nn.SpatialBatchNormalization(128)(cnv2_))
    local r3 = Residual(128, 128)(cnv2)
    local pool2 = cudnn.SpatialMaxPooling(2, 2, 2, 2)(r3) -- 32
    local r4 = Residual(128,128)(pool2)

    local cnv3_ = cudnn.SpatialConvolution(128,128,3,3,1,1,1,1)(r4)
    local cnv3 = cudnn.ReLU(true)(nn.SpatialBatchNormalization(128)(cnv3_))
    local r5 = Residual(128, 128)(cnv3)
    local pool3 = cudnn.SpatialMaxPooling(2, 2, 2, 2)(r5) -- 16
    local r6 = Residual(128, 128)(pool3)

    local cnv4_ = cudnn.SpatialConvolution(128,128,3,3,1,1,1,1)(r6)
    local cnv4 = cudnn.ReLU(true)(nn.SpatialBatchNormalization(128)(cnv4_))
    local r7 = Residual(128, 128)(cnv4)
    local pool4 = cudnn.SpatialMaxPooling(2, 2, 2, 2)(r7)  -- 8
    local r8 = Residual(128, 128)(pool4)

    local cnv5_ = cudnn.SpatialConvolution(128,128,3,3,1,1,1,1)(r8)
    local cnv5 = cudnn.ReLU(true)(nn.SpatialBatchNormalization(128)(cnv5_))
    local r9 = Residual(128, 128)(cnv5)
    local pool5 = cudnn.SpatialMaxPooling(2, 2, 2, 2)(r9)  -- 4
    local r10 = Residual(128, 128)(pool5)

    local r17 = linVector(128 * 16,opt.nFeats * 4, nn.View(-1, 128 * 16):cuda()(r10)) -- 2048 -> 1024

    -- Residual layers at output resolution
    local ll = r17
    for j = 1,opt.nModules do ll = linVector(opt.nFeats * 4,opt.nFeats * 2, ll) end -- 1024 -> 512
    -- Linear layer to produce first set of predictions
    ll = linVector(opt.nFeats*2,opt.nFeats,ll) -- 512 -> 256

    -- Predicted heatmaps
    local out = nn.Linear(opt.nFeats,opt.nOutChannels):cuda()(ll) -- 256 -> 10

    -- Final model
    local model = nn.gModule({inp}, {out})

    print('Img -> fc network')
    return model

end