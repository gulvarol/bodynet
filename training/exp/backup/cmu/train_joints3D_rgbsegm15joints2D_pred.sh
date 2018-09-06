cd ..
    qlua main.lua \
    -dirName joints3D/rgbsegm15joints2D/pred \
    -input rgb -applyHG rgbsegm15joints2D \
    -supervision joints3D \
    -datasetname cmu \
-batchSize 6 \

