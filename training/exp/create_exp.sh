#!/bin/bash
#
# Example usage:
#
#       input="rgb"
#       supervision="segm15joints2Djoints3Dvoxels" 
#       inputtype="gt"
#       extra_args="_FVSV"
#       running_mode="train"
#       dataset="cmu"
#
#       create_cmd
#       cmd="${return_str} \\
#       -batchSize 4 \\
#       -proj silhFVSV \\
#       -modelVoxels models/t7/model_voxels_FVSV_cmu.t7 \\
#       "
#       run_cmd

help_str="
This is a script common to all experiments, it

 (1) creates automatically the command string given input/output of the network,
 (2) creates a file with the command string

Set the following variables in your experiment script:

 input       : Input of the network (rgb | segm | joints2D | segm15joints2D | rgbsegm15joints2D | rgbsegm15joints2Djoints3D)
 supervision : Output of the network (segm | joints2D | joints3D | segm15joints2Djoints3D | segm15joints2Djoints3Dvoxels | pose | shape | poseshape)
 inputtype   : Whether the network will be trained with ground truth only, predictions only or both together, make sure to set gt for rgb input.(gt | pred | mix)
 extra_args  : Any extra string that will be concatenated to the experiment name. (e.g., if you modify the $cmd string with extra arguments such as -proj, add _proj)


A lot of variables are global, use with caution. An example usage is included in comment.
"

if [ $1 ] && [ $1  = "-h" ]
then
    echo "${help_str}"
else
    echo
    echo "Type to see help message: ./create_exp.sh -h"
    echo
fi

pretty_print() {
    echo "----------------------------------------------------------------------------------------------"
    echo 
    echo "=====> $1:"
    echo 
    echo "$2"
    echo
    return 0
}

create_cmd() {
    running_mode=${running_mode:-'train'}
    dataset=${dataset:-'cmu'}
    #modelno

    # Whether the network will be trained with predictions (pred), ground truth (gt) or mixture of the two (mix) as input
    if [ $inputtype = "pred" ]
    then
        input_str="-input rgb -applyHG ${input}"
    elif [ $inputtype = "gt" ]
    then
        input_str="-input ${input}"
    elif [ $inputtype = "mix" ]
    then
        input_str="-input rgb -applyHG ${input} -mix"
    fi


    uniquename="${running_mode}_${supervision}_${input}_${inputtype}${extra_args}"
    expdir="${supervision}/${input}/${inputtype}${extra_args}" # e.g. joints3D/rgbsegm15joints2D/pred

    if [ $running_mode == "test" ]
    then
        expdirroot="${expdir}"
        expdir="${expdir}/test_${modelno}"
    elif [ $running_mode == "vis" ]
    then
        expdirroot="${expdir}"
        expdir="vis"
    fi

    expdirfull="${HOME}/cnn_saves/${dataset}/${expdir}" # e.g. /home/gvarol/cnn_saves/cmu/joints3D/rgbsegm15joints2D/pred
    pretty_print "Mkdir" ${expdirfull}
    mkdir -p ${expdirfull}

    return_str=${return_str}"cd ..
    qlua main.lua \\
    -dirName ${expdir} \\
    ${input_str} \\
    -supervision ${supervision} \\
    -datasetname ${dataset}"

    # Whether network will be trained (train), evaluated (test) or visualized (vis)
    if [ $running_mode = "test" ]
    then
        return_str=${return_str}"\\
        -evaluate\\
        -retrain ${HOME}/cnn_saves/${dataset}/${expdirroot}/model_${modelno}.t7"
    elif [ $running_mode = "vis" ]
    then
        return_str=${return_str}"\\
        -show -nDonkeys 0 -batchSize 1\\
        -retrain ${HOME}/cnn_saves/${dataset}/${expdirroot}/model_${modelno}.t7"
    fi
}

cp_script_files() {
    scriptfile1="${expdirfull}/auto_run.sh"
    if [ -f ${scriptfile1} ]; then
        echo "${scriptfile1} exists."
        echo ""
        echo "You can:"
        echo "rm ${scriptfile1}"
        echo ""
        exit
    fi

    pretty_print "Creating script file1 (run)" ${scriptfile1}
    echo "${cmd}" > ${scriptfile1} # quotes are important to get linebreaks

    scriptfile2="backup/${dataset}/${uniquename}.sh"
    pretty_print "Creating script file2 (backup)" ${scriptfile2}
    echo "${cmd}" > ${scriptfile2}
}

run_cmd() {
    cp_script_files
    pretty_print "Running" "${cmd}"
    bash ${scriptfile1}
}