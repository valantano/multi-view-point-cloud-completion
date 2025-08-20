#!/usr/bin/zsh

if [ "$USER" = "valantano" ]; then
    SCRIPT_DIR=$(dirname "$(realpath "$0")")
else
    SCRIPT_DIR='/home/fn848825/multi-view-point-cloud-completion/base/scripts/'
fi



# bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh CompletionPoseEstCfgs CompletionPoseEst_default_volar.yaml FFPCompDefaultV0 FFPCDefaultV False
# bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh CompletionPoseEstCfgs CompletionPoseEst_default_volar_input.yaml FFPCompDefaultInputV0 FFPCDefaultInputV False
bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh CompletionPoseEstCfgs CompletionPoseEst_default_volar_concat.yaml FFPCompPrePeter_ConcatV0 FFPCConcatV False
# bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh CompletionPoseEstCfgs CompletionPoseEst_default_volar_concat_input.yaml FFPCompConcatInputV0 FFPCConcatInputV False
bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh CompletionPoseEstCfgs CompletionPoseEst_default_volar_affil.yaml FFPCompPrePeter_AffilV0 FFPCPreAffilV False
# bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh CompletionPoseEstCfgs CompletionPoseEst_default_volar_affil_input.yaml FFPCompAffilInputV0 FFPCAffilInputV False

# bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh CompletionPoseEstCfgs CompletionPoseEst_default_dorsal.yaml FFPCompDefaultD0 FFPCDefaultD False
# bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh CompletionPoseEstCfgs CompletionPoseEst_default_dorsal_input.yaml FFPCompDefaultInputD0 FFPCDefaultInputD False
bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh CompletionPoseEstCfgs CompletionPoseEst_default_dorsal_concat.yaml FFPCompPrePeter_ConcatD0 FFPCConcatD False
# bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh CompletionPoseEstCfgs CompletionPoseEst_default_dorsal_concat_input.yaml FFPCompConcatInputD0 FFPCConcatInputD False
bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh CompletionPoseEstCfgs CompletionPoseEst_default_dorsal_affil.yaml FFPCompPrePeter_AffilD0 FFPCAffilD False
# bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh CompletionPoseEstCfgs CompletionPoseEst_default_dorsal_affil_input.yaml FFPCompAffilInputD0 FFPCAffilInputD False