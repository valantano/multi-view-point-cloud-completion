#!/usr/bin/zsh

if [ "$USER" = "valantano" ]; then
    SCRIPT_DIR=$(dirname "$(realpath "$0")")
else
    SCRIPT_DIR='/home/fn848825/multi-view-point-cloud-completion/base/scripts/'
fi


bash ${SCRIPT_DIR}/experiment_pose_pointattn.sh PoseEstCfgs PoseEst_ScaphoidPointAttN_volar.yaml FPoseVolarPre0 LowPoseV False
bash ${SCRIPT_DIR}/experiment_pose_pointattn.sh PoseEstCfgs PoseEst_ScaphoidPointAttN_dorsal.yaml FPoseDorsalPre0 LowPoseD False
# bash ${SCRIPT_DIR}/experiment_pose_pointattn.sh PoseEstCfgs PCD_Aligner.yaml 1PoseAligner PoseA False


