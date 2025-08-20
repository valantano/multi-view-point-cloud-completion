#!/usr/bin/zsh

if [ "$USER" = "valantano" ]; then
    SCRIPT_DIR=$(dirname "$(realpath "$0")")
else
    SCRIPT_DIR='/home/fn848825/multi-view-point-cloud-completion/base/scripts/'
fi



# bash ${SCRIPT_DIR}/experiment_rotation_pointattn.sh RotationCfgs ScaphoidPointAttN_rotation_RERG_dorsal.yaml +RERGDorsalS DRERG True dorsal
# bash ${SCRIPT_DIR}/experiment_rotation_pointattn.sh RotationCfgs ScaphoidPointAttN_rotation_RERG_dorsal.yaml +RERGDorsal DRERGDorsal False dorsal


# bash ${SCRIPT_DIR}/experiment_rotation_pointattn.sh RotationCfgs ScaphoidPointAttN_rotation_RERG_volar.yaml +RERGVolarS VRERG True volar
# bash ${SCRIPT_DIR}/experiment_rotation_pointattn.sh RotationCfgs ScaphoidPointAttN_rotation_RERG_volar.yaml +RERGVolar DRERGVolar False volar


bash ${SCRIPT_DIR}/experiment_rotation_pointattn.sh RotationCfgs ScaphoidPointAttN_rotation_RERG_volar_hyp.yaml +RERGVolarLR VRERG False volar


