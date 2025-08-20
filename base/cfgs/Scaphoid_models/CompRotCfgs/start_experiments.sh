#!/usr/bin/zsh

if [ "$USER" = "valantano" ]; then
    SCRIPT_DIR=/home/valantano/mt/repository/base/scripts/
else
    SCRIPT_DIR='/home/fn848825/multi-view-point-cloud-completion/base/scripts/'
fi



bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh CompRotCfgs ScaphoidPointAttN_rotation_simple_alignment_dorsal.yaml +CRERGDorsalPreS DRERG True
# bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh CompRotCfgs ScaphoidPointAttN_rotation_simple_alignment_dorsal.yaml +CRERGDorsalPre DRERG False

# bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh CompRotCfgs ScaphoidPointAttN_rotation_simple_alignment_volar.yaml +CRERGVolarS VRERG True
# bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh CompRotCfgs ScaphoidPointAttN_rotation_simple_alignment_volar.yaml +CRERGVolar vRERG False

