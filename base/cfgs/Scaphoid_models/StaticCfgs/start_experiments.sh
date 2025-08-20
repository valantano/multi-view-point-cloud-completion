#!/usr/bin/zsh

if [ "$USER" = "valantano" ]; then
    SCRIPT_DIR=/home/valantano/mt/repository/base/scripts/
else
    SCRIPT_DIR='/home/fn848825/multi-view-point-cloud-completion/base/scripts/'
fi

CONFIG_FOLDER=StaticCfgs

# bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh ${CONFIG_FOLDER} ScaphoidPointAttN_Baseline_Max_static.yaml X600StaticBaseMaxPre0 MaxStatic
# bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh ${CONFIG_FOLDER} ScaphoidPointAttN_Baseline_Min_dorsal_static.yaml X600StaticDorsalPre0 DorsalStatic
# bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh ${CONFIG_FOLDER} ScaphoidPointAttN_Baseline_Min_volar_static.yaml X600StaticVolarPre0 VolarStatic


# bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh ${CONFIG_FOLDER} ScaphoidPointAttN_Both_Affil.yaml X600StaticAffil0 StaticAffil
# bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh ${CONFIG_FOLDER} ScaphoidPointAttN_Both_Concat.yaml X600StaticConcat0 StaticConcat
# bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh ${CONFIG_FOLDER} ScaphoidPointAttN_Both_Double_Net.yaml X600StaticDoubleNet0 StaticDoubleNet


# bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh ${CONFIG_FOLDER} ScaphoidPointAttN_Both_Affil_Experimental.yaml ExperimentalStaticAffil0 StaticAffil
# bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh ${CONFIG_FOLDER} ScaphoidPointAttN_Both_Concat_Experimental.yaml ExperimentalStaticConcat0 StaticConcat

bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh ${CONFIG_FOLDER} ScaphoidPointAttN_Both_Concat_Null.yaml XStaticConcatNull0 NullStaticConcat
bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh ${CONFIG_FOLDER} ScaphoidPointAttN_Both_Concat_Volar.yaml XStaticConcatVolar0 VolarStaticConcat
bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh ${CONFIG_FOLDER} ScaphoidPointAttN_Both_Concat_Dorsal.yaml XStaticConcatDorsal0 DorsalStaticConcat
bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh ${CONFIG_FOLDER} ScaphoidPointAttN_Both_Concat_Concat.yaml XStaticConcatConcat0 BothStaticConcat

bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh ${CONFIG_FOLDER} ScaphoidPointAttN_Both_Affil_Null.yaml XStaticAffilNull0 NullStaticAffil
bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh ${CONFIG_FOLDER} ScaphoidPointAttN_Both_Affil_Volar.yaml XStaticAffilVolar0 VolarStaticAffil
bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh ${CONFIG_FOLDER} ScaphoidPointAttN_Both_Affil_Dorsal.yaml XStaticAffilDorsal0 DorsalStaticAffil
bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh ${CONFIG_FOLDER} ScaphoidPointAttN_Both_Affil_Affil.yaml XStaticAffilAffil0 BothStaticAffil