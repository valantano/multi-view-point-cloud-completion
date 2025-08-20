#!/usr/bin/zsh

#SBATCH --job-name=MaxV
#SBATCH -A thes1928


#SBATCH --time=0-08:00              			# Runtime in D-HH:MM
#SBATCH --output=output.%J.log         			# File to which STDOUT will be written
#SBATCH --mail-type=ALL         			# Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=valentino.geuenich@rwth-aachen.de     	# Email to which notifications will be sent
#SBATCH --gres=gpu:1
#SBATCH --mem=10G

module restore mt9

source ~/.bashrc


# if user is not valantano then use second script_dir
if [ "$USER" = "valantano" ]; then
    SCRIPT_DIR=$(dirname "$(realpath "$0")")
else
    SCRIPT_DIR='/home/fn848825/multi-view-point-cloud-completion/base/scripts/'
fi


cd "$SCRIPT_DIR"
cd "../"

CONFIG_FOLDER=${SCRIPT_DIR}/../cfgs/

FOLDER_NAME=BaselineCfgs
CONFIG_NAME=ScaphoidPointAttN_Baseline_Max.yaml
# CONFIG_NAME=ScaphoidPointAttN_Baseline_Max_old_hyps.yaml
# CONFIG_NAME=ScaphoidPointAttN_Baseline_Min_dorsal.yaml
# CONFIG_NAME=ScaphoidPointAttN_Baseline_Min_volar.yaml

# FOLDER_NAME=StaticCfgs
# CONFIG_NAME=ScaphoidPointAttN_Baseline_Min_dorsal_static.yaml
# CONFIG_NAME=ScaphoidPointAttN_Baseline_Min_volar_static.yaml
# CONFIG_NAME=ScaphoidPointAttN_Baseline_Max_static.yaml
# CONFIG_NAME=ScaphoidPointAttN_Double_Net.yaml

# FOLDER_NAME=SeedAttnMatcherCfgs
# CONFIG_NAME=ScaphoidPointAttN_SeedAttnMatcher_double_volar.yaml
# CONFIG_NAME=ScaphoidPointAttN_SeedAttnMatcher_double_dorsal.yaml
# CONFIG_NAME=ScaphoidPointAttN_SeedAttnMatcher_volar.yaml
# CONFIG_NAME=ScaphoidPointAttN_SeedAttnMatcher_dorsal.yaml

# FOLDER_NAME=AffiliationCfgs
# CONFIG_NAME=ScaphoidPointAttN_affil_dorsal.yaml

# FOLDER_NAME=NullSeeds
# CONFIG_NAME=ScaphoidPointAttN_Baseline_Max.yaml

# FOLDER_NAME=ConcatCfgs
# CONFIG_NAME=ScaphoidPointAttN_concat_dorsal.yaml
# # CONFIG_NAME=ScaphoidPointAttN_concat_volar.yaml
# # CONFIG_NAME=ScaphoidPointAttN_concat_static.yaml
# # CONFIG_NAME=ScaphoidPointAttN_fuser_dorsal.yaml
# # CONFIG_NAME=ScaphoidPointAttN_concat_control_dorsal.yaml


# FOLDER_NAME=CompletionPoseEstCfgs
# CONFIG_NAME=CompletionPoseEst_ScaphoidPointAttN_dorsal.yaml
# # # CONFIG_NAME=CompletionPoseEst_ScaphoidPointAttN_volar.yaml
# CONFIG_NAME=CompletionPoseEst_concat_input_dorsal.yaml



FOLDER_NAME_LIST=(StaticCfgs StaticCfgs StaticCfgs StaticCfgs StaticCfgs StaticCfgs)
CONFIG_NAME_LIST=(ScaphoidPointAttN_Both_Concat_Concat.yaml ScaphoidPointAttN_Both_Affil.yaml ScaphoidPointAttN_Baseline_Min_volar_static.yaml ScaphoidPointAttN_Baseline_Min_dorsal_static.yaml ScaphoidPointAttN_Baseline_Max_static.yaml ScaphoidPointAttN_Both_Double_Net.yaml)

EXP_NAME_LIST=(XStaticConcatConcat0) # X600StaticAffil0) # X600StaticVolar0 X600StaticDorsal0 X600StaticBaseMax0 X600StaticDoubleNet0 )


# bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh CompletionPoseEstCfgs CompletionPoseEst_default_volar.yaml FPCompDefaultV0 FPCDefaultV False
# # bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh CompletionPoseEstCfgs CompletionPoseEst_default_volar_input.yaml FPCompDefaultInputV0 FPCDefaultInputV False
# bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh CompletionPoseEstCfgs CompletionPoseEst_default_volar_concat.yaml FPCompConcatV0 FPCConcatV False
# # bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh CompletionPoseEstCfgs CompletionPoseEst_default_volar_concat_input.yaml FPCompConcatInputV0 FPCConcatInputV False
# bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh CompletionPoseEstCfgs CompletionPoseEst_default_volar_affil.yaml FPCompAffilV0 FPCAffilV False
# # bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh CompletionPoseEstCfgs CompletionPoseEst_default_volar_affil_input.yaml FPCompAffilInputV0 FPCAffilInputV False

# bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh CompletionPoseEstCfgs CompletionPoseEst_default_dorsal.yaml FPCompDefaultD0 FPCDefaultD False
# # bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh CompletionPoseEstCfgs CompletionPoseEst_default_dorsal_input.yaml FPCompDefaultInputD0 FPCDefaultInputD False
# bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh CompletionPoseEstCfgs CompletionPoseEst_default_dorsal_concat.yaml FPCompConcatD0 FPCConcatD False
# # bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh CompletionPoseEstCfgs CompletionPoseEst_default_dorsal_concat_input.yaml FPCompConcatInputD0 FPCConcatInputD False
# bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh CompletionPoseEstCfgs CompletionPoseEst_default_dorsal_affil.yaml FPCompAffilD0 FPCAffilD False
# bash ${SCRIPT_DIR}/experiment_scaphoid_pointattn.sh CompletionPoseEstCfgs CompletionPoseEst_default_dorsal_affil_input.yaml FPCompAffilInputD0 FPCAffilInputD False

FOLDER_NAME_LIST=(CompletionPoseEstCfgs CompletionPoseEstCfgs CompletionPoseEstCfgs CompletionPoseEstCfgs CompletionPoseEstCfgs CompletionPoseEstCfgs)
CONFIG_NAME_LIST=(CompletionPoseEst_default_volar.yaml CompletionPoseEst_default_volar_concat.yaml CompletionPoseEst_default_dorsal_concat.yaml CompletionPoseEst_default_volar_affil.yaml CompletionPoseEst_default_dorsal.yaml CompletionPoseEst_default_dorsal_affil.yaml)
EXP_NAME_LIST=(FPCompDefaultV0 FPCompConcatV0 FPCompConcatD0 FPCompAffilV0 FPCompDefaultD0 FPCompAffilD0)


# FOLDER_NAME_LIST=(BaselineCfgs BaselineCfgs BaselineCfgs BaselineCfgs BaselineCfgs BaselineCfgs)
# CONFIG_NAME_LIST=(ScaphoidPointAttN_Baseline_Max.yaml ScaphoidPointAttN_Baseline_Min_dorsal.yaml ScaphoidPointAttN_Baseline_Min_volar.yaml ScaphoidPointAttN_Baseline_Max.yaml ScaphoidPointAttN_Baseline_Min_dorsal.yaml ScaphoidPointAttN_Baseline_Min_volar.yaml)
# EXP_NAME_LIST=(FBaseMax0 FBaseMinDorsal0 FBaseMinVolar0 FBaseMaxPre0 FBaseMinDorsalPre0 FBaseMinVolarPre0)

# FOLDER_NAME_LIST=(AdaPoinTr AdaPoinTr AdaPoinTr)
# CONFIG_NAME_LIST=(ScaphoidAdaPoinTr_Min_dorsal.yaml ScaphoidAdaPoinTr_Min_volar.yaml ScaphoidAdaPoinTr_Max.yaml)
# EXP_NAME_LIST=(FAdaMinPreD0 FAdaMinPreV0 FAdaMaxPre0)

# FOLDER_NAME_LIST=(BaselineCfgs BaselineCfgs BaselineCfgs BaselineCfgs)
# CONFIG_NAME_LIST=(ScaphoidPointAttN_Baseline_Max.yaml ScaphoidPointAttN_Baseline_Min_dorsal.yaml ScaphoidPointAttN_Baseline_Min_volar.yaml)
# EXP_NAME_LIST=(FBaseMaxPre0 FBaseMinDorsalPre0 FBaseMinVolarPre0)

# FOLDER_NAME_LIST=(AffiliationCfgs AffiliationCfgs ConcatCfgs ConcatCfgs)
# CONFIG_NAME_LIST=(ScaphoidPointAttN_affil_dorsal.yaml ScaphoidPointAttN_affil_volar.yaml ScaphoidPointAttN_concat_dorsal.yaml ScaphoidPointAttN_concat_volar.yaml)
# EXP_NAME_LIST=(FAffilDorsal0 FAffilVolar0 FConcatDorsalSingle0 FConcatVolarSingle0)

# FOLDER_NAME_LIST=(ConcatCfgs ConcatCfgs)
# CONFIG_NAME_LIST=(ScaphoidPointAttN_concat_dorsal.yaml ScaphoidPointAttN_concat_volar.yaml)
# # EXP_NAME_LIST=(FConcatDorsalSingle0 FConcatVolarSingle0)
# EXP_NAME_LIST=(FConcatDorsalSingle0) # FConcatVolarsolo0)

# FOLDER_NAME_LIST=(CompletionPoseEstCfgs CompletionPoseEstCfgs CompletionPoseEstCfgs CompletionPoseEstCfgs) # CompletionPoseEstCfgs CompletionPoseEstCfgs)
# CONFIG_NAME_LIST=(CompletionPoseEst_default_volar_concat.yaml CompletionPoseEst_default_dorsal_concat.yaml CompletionPoseEst_default_dorsal_affil.yaml CompletionPoseEst_default_volar_affil.yaml)
# EXP_NAME_LIST=(FFPCompPrePeter_ConcatV0 FFPCompPrePeter_ConcatD0 FFPCompPrePeter_AffilD0  FFPCompPrePeter_AffilV0)



#################### AdaPoinTr Configs #####################################################
# FOLDER_NAME=AdaPoinTr
# CONFIG_NAME=ScaphoidAdaPoinTr_Max.yaml

# PRETRAINED_WEIGHTS=${SCRIPT_DIR}/../pretrained_weights/AdaPoinTr_ps34.pth
############################################################################################


# FOLDER_NAME_LIST=(StaticCfgs CompletionPoseEstCfgs CompletionPoseEstCfgs AdaPoinTr AdaPoinTr AdaPoinTr BaselineCfgs BaselineCfgs BaselineCfgs)
# CONFIG_NAME_LIST=(ScaphoidPointAttN_Both_Concat_Concat.yaml CompletionPoseEst_default_volar_concat.yaml CompletionPoseEst_default_dorsal_concat.yaml ScaphoidAdaPoinTr_Min_dorsal.yaml ScaphoidAdaPoinTr_Min_volar.yaml ScaphoidAdaPoinTr_Max.yaml ScaphoidPointAttN_Baseline_Max.yaml ScaphoidPointAttN_Baseline_Min_dorsal.yaml ScaphoidPointAttN_Baseline_Min_volar.yaml)

# EXP_NAME_LIST=(XStaticConcatConcat0 FFPCompPrePeter_ConcatV0 FFPCompPrePeter_ConcatD0 FAdaMinPreD0 FAdaMinPreV0 FAdaMaxPre0 FBaseMaxPre0 FBaseMinDorsalPre0 FBaseMinVolarPre0) # X600StaticAffil0) # X600StaticVolar0 X600StaticDorsal0 X600StaticBaseMax0 X600StaticDoubleNet0 )


CONFIG=Scaphoid_models/${FOLDER_NAME}/${CONFIG_NAME}


# PRETRAINED_WEIGHTS=${SCRIPT_DIR}/../pretrained_weights/pointattn_c3d.pth
PRETRAINED_WEIGHTS=${SCRIPT_DIR}/../pretrained_weights/pointattn_pcn.pth

for i in {0..10}; do
    FOLDER_NAME=${FOLDER_NAME_LIST[$i]}
    CONFIG_NAME=${CONFIG_NAME_LIST[$i]}
    EXP_NAME=${EXP_NAME_LIST[$i]}
    CONFIG=Scaphoid_models/${FOLDER_NAME}/${CONFIG_NAME}

    bash ./scripts/__train.sh 0 \
        --config_folder ${CONFIG_FOLDER} \
        --config ${CONFIG} \
        --exp_name ${EXP_NAME} \
        --strict False \
        --test \
        "$@"
done