#!/usr/bin/zsh

if [ "$USER" = "valantano" ]; then
    SCRIPT_DIR=/home/valantano/mt/repository/base/scripts/
else
    SCRIPT_DIR='/home/fn848825/multi-view-point-cloud-completion/base/scripts/'
fi

# bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh BaselineCfgs ScaphoidPointAttN_Baseline_Max.yaml +BaseMax Max
# bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh BaselineCfgs ScaphoidPointAttN_Baseline_Min_dorsal.yaml +BaseMinDW2 MinDorsalW2
# bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh BaselineCfgs ScaphoidPointAttN_Baseline_Min_volar.yaml +BaseMinVW2 MinVolarW2

CONFIG_FOLDER=SeedAttnMatcherCfgs

# bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh ${CONFIG_FOLDER} ScaphoidPointAttN_SeedAttnMatcher_volar.yaml +SAMV SAMV
# bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh ${CONFIG_FOLDER} ScaphoidPointAttN_SeedAttnMatcher_double_volar.yaml +SAMdoubleV SAMdoubleV

bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh ${CONFIG_FOLDER} ScaphoidPointAttN_SeedAttnMatcher_dorsal.yaml +SAMDPart SAMdPart
bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh ${CONFIG_FOLDER} ScaphoidPointAttN_SeedAttnMatcher_double_dorsal.yaml +SAMdoubleDPart SAMdoubleD


# bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh ${CONFIG_FOLDER} ScaphoidPointAttN_Baseline_Min_dorsal.yaml +BaseMinDW2 MinDorsalW2
# bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh ${CONFIG_FOLDER} ScaphoidPointAttN_Baseline_Min_volar.yaml +BaseMinVW2 MinVolarW2