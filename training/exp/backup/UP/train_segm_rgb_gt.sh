cd ..
    qlua main.lua \
    -dirName segm/rgb/gt \
    -input rgb \
    -supervision segm \
    -datasetname UP \
-batchSize 12 \
-retrain models/t7/model_segm_cmu.t7 \
-testDir test_lsp \

