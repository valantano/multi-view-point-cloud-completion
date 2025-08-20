#!/usr/bin/zsh

if [ "$USER" = "valantano" ]; then
    SCRIPT_DIR=$(dirname "$(realpath "$0")")
else
    SCRIPT_DIR='/home/fn848825/multi-view-point-cloud-completion/base/scripts/'
fi


bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh AdaPoinTr ScaphoidAdaPoinTr_Min_dorsal.yaml FAdaMinPreD0 AdaMinD False
bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh AdaPoinTr ScaphoidAdaPoinTr_Min_volar.yaml FAdaMinPreV0 AdaMinV False
# bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh AdaPoinTr ScaphoidAdaPoinTr_Max.yaml FAdaMaxPre0 AdaMax False


