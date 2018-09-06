cd ..
    qlua main.lua \
    -dirName segm15joints2Djoints3Dvoxels/rgb/gt_FVSV \
    -input rgb \
    -supervision segm15joints2Djoints3Dvoxels \
    -datasetname cmu \
-batchSize 4 \
-proj silhFVSV \
-modelVoxels models/t7/model_voxels_FVSV_cmu.t7 \

