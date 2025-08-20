#!/usr/bin/zsh

if [ "$USER" = "valantano" ]; then
    SCRIPT_DIR=$(dirname "$(realpath "$0")")
else
    SCRIPT_DIR='/home/fn848825/multi-view-point-cloud-completion/base/scripts/'
fi

bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh BaselineCfgs ScaphoidPointAttN_Baseline_Max.yaml +BaseMaxNull MaxNull