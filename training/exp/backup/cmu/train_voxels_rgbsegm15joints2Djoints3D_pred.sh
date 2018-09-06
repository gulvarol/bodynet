cd ..
    qlua main.lua \
    -dirName voxels/rgbsegm15joints2Djoints3D/pred \
    -input rgb -applyHG rgbsegm15joints2Djoints3D \
    -supervision voxels \
    -datasetname cmu \
-batchSize 7 \

