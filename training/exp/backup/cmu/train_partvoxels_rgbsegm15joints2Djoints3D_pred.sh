cd ..
    qlua main.lua \
    -dirName partvoxels/rgbsegm15joints2Djoints3D/pred \
    -input rgb -applyHG rgbsegm15joints2Djoints3D \
    -supervision partvoxels \
    -datasetname cmu \
-batchSize 4 \
-retrain models/t7/init_partvoxels.t7 \

