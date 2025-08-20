#!/usr/bin/zsh

if [ "$USER" = "valantano" ]; then
    SCRIPT_DIR=$(dirname "$(realpath "$0")")
else
    SCRIPT_DIR='/home/fn848825/multi-view-point-cloud-completion/base/scripts/'
fi

bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh BaselineCfgs ScaphoidPointAttN_Baseline_Max.yaml FBaseMaxPre0 Max
bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh BaselineCfgs ScaphoidPointAttN_Baseline_Min_dorsal.yaml FBaseMinDorsalPre0 MinDorsal
bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh BaselineCfgs ScaphoidPointAttN_Baseline_Min_volar.yaml FBaseMinVolarPre0 MinVolar

# bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh BaselineCfgs ScaphoidPointAttN_Baseline_Max.yaml +BaseMaxW2 MaxW2
# bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh BaselineCfgs ScaphoidPointAttN_Baseline_Min_dorsal.yaml +BaseMinDW2 MinDorsalW2
# bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh BaselineCfgs ScaphoidPointAttN_Baseline_Min_volar.yaml +BaseMinVW2 MinVolarW2