cd ..
    qlua main.lua \
    -dirName voxels/rgbsegm15joints2Djoints3D/pred_FVSV \
    -input rgb -applyHG rgbsegm15joints2Djoints3D \
    -supervision voxels \
    -datasetname cmu \
-batchSize 7 \
-proj silhFVSV \
-modelVoxels models/t7/model_voxels_cmu.t7 \

