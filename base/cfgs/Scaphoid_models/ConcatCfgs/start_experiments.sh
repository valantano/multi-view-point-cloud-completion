#!/usr/bin/zsh

if [ "$USER" = "valantano" ]; then
    SCRIPT_DIR=$(dirname "$(realpath "$0")")
else
    SCRIPT_DIR='/home/fn848825/multi-view-point-cloud-completion/base/scripts/'
fi

CONFIG_FOLDER=ConcatCfgs

# bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh ${CONFIG_FOLDER} ScaphoidPointAttN_concat_dorsal.yaml FConcatDorsalSolo0 ConcatD
bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh ${CONFIG_FOLDER} ScaphoidPointAttN_concat_volar.yaml FConcatVolarsolo0 ConcatV
# bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh ${CONFIG_FOLDER} ScaphoidPointAttN_concat_static.yaml +ConcatStatic ConcatStatic
