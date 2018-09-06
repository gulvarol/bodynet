#!/bin/bash
source create_exp.sh -h

input="rgb"
supervision="segm15joints2Djoints3Dvoxels" 
inputtype="gt"
extra_args="_FVSV"
running_mode="train"
#modelno=1
dataset="cmu"

create_cmd
cmd="${return_str} \\
-batchSize 4 \\
-proj silhFVSV \\
-modelVoxels models/t7/model_voxels_FVSV_cmu.t7 \\
"
run_cmd