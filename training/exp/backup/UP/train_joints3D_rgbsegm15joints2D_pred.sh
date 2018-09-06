cd ..
    qlua main.lua \
    -dirName joints3D/rgbsegm15joints2D/pred \
    -input rgb -applyHG rgbsegm15joints2D \
    -supervision joints3D \
    -datasetname UP \
-batchSize 6 \
-retrain models/t7/model_joints3D_cmu.t7 \
-testDir test_lsp \

