function createModel()

    print('Loading pre-trained segmentation network: ' .. opt.modelSegm)
    local modelSegm     = torch.load(opt.modelSegm)
    print('Loading pre-trained joints2D network: ' .. opt.modelJoints2D)
    local modelJoints2D = torch.load(opt.modelJoints2D)
    print('Loading pre-trained joints3D network: ' .. opt.modelJoints3D)
    local modelJoints3D = torch.load(opt.modelJoints3D)
    print('Loading pre-trained voxels network: ' .. opt.modelVoxels)
    local modelVoxels   = torch.load(opt.modelVoxels)

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
    --local joints3Dout = createJoints3D()(rgbsegm15joints2DInp) -- 8 x 65*24 x 64 x 64

    local joints3DInp = nn.SelectTable(opt.nStackJoints3D)(joints3DOut)

    local voxelsOut = (modelVoxels)({rgbsegm15joints2DInp, joints3DInp})

    local out = {}
    for st = 1, opt.nStack do
        table.insert(out, nn.SelectTable(st)(segmOut))
        table.insert(out, nn.SelectTable(st)(joints2DOut))
        table.insert(out, nn.SelectTable(st)(joints3DOut))
        if (opt.proj == 'silhFV' or opt.proj == 'segmFV') then
            table.insert(out, nn.SelectTable(2*st-1)(voxelsOut))
            table.insert(out, nn.SelectTable(2*st)(voxelsOut))
        elseif(opt.proj == 'silhFVSV' or opt.proj == 'segmFVSV') then
            table.insert(out, nn.SelectTable(3*st-2)(voxelsOut))
            table.insert(out, nn.SelectTable(3*st-1)(voxelsOut))
            table.insert(out, nn.SelectTable(3*st)(voxelsOut))
        else
            table.insert(out, nn.SelectTable(st)(voxelsOut))
        end
    end

    local model = nn.gModule({rgb}, out)
    print('Return end-to-end rgb-> joints2D, segm -> joints3D -> voxels network with ' .. opt.proj)
    return model
end




