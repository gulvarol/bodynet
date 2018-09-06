cd ..
    qlua main.lua \
    -dirName partvoxels/rgbsegm15joints2Djoints3D/pred_FVSV \
    -input rgb -applyHG rgbsegm15joints2Djoints3D \
    -supervision partvoxels \
    -datasetname cmu \
-batchSize 2 \
-proj segmFVSV \
-modelVoxels /home/gvarol/cnn_saves/cmu/partvoxels/rgbsegm15joints2Djoints3D/pred_pretrained/model_29.t7 \

