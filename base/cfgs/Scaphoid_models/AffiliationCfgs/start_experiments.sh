#!/usr/bin/zsh

if [ "$USER" = "valantano" ]; then
    SCRIPT_DIR=$(dirname "$(realpath "$0")")
else
    SCRIPT_DIR='/home/fn848825/multi-view-point-cloud-completion/base/scripts/'
fi


CONFIG_FOLDER=AffiliationCfgs

bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh ${CONFIG_FOLDER} ScaphoidPointAttN_affil_dorsal.yaml FAffilDorsal0 AffDorsal
bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh ${CONFIG_FOLDER} ScaphoidPointAttN_affil_volar.yaml FAffilVolar0 AffVolar

# bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh ${CONFIG_FOLDER} ScaphoidPointAttN_affil_static.yaml +AffilStatic600 AffStatic600
