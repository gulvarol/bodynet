cd ..
    qlua main.lua \
    -dirName segm15joints2Djoints3Dvoxels/rgb/gt_FVSV \
    -input rgb \
    -supervision segm15joints2Djoints3Dvoxels \
    -datasetname UP \
-batchSize 4 \
-proj silhFVSV \
-modelVoxels models/t7/model_voxels_FVSV_UP.t7 \
-testDir test_lsp \

