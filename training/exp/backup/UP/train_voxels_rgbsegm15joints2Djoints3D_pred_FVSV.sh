cd ..
    qlua main.lua \
    -dirName voxels/rgbsegm15joints2Djoints3D/pred_FVSV \
    -input rgb -applyHG rgbsegm15joints2Djoints3D \
    -supervision voxels \
    -datasetname UP \
-batchSize 6 \
-proj silhFVSV \
-retrain models/t7/model_voxels_FVSV_cmu.t7 \
-testDir test_lsp \

