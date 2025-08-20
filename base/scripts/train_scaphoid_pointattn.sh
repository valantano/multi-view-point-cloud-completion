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
CONFIG_NAME=ScaphoidPointAttN_Baseline_Min_dorsal.yaml
# CONFIG_NAME=ScaphoidPointAttN_Baseline_Min_volar.yaml

# FOLDER_NAME=StaticCfgs
# CONFIG_NAME=ScaphoidPointAttN_Baseline_Min_dorsal_static.yaml
# CONFIG_NAME=ScaphoidPointAttN_Baseline_Min_volar_static.yaml
# CONFIG_NAME=ScaphoidPointAttN_Baseline_Max_static.yaml
# CONFIG_NAME=ScaphoidPointAttN_Double_Net.yaml
# CONFIG_NAME=ScaphoidPointAttN_Both_Concat.yaml
# CONFIG_NAME=ScaphoidPointAttN_Both_Affil.yaml


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



#################### AdaPoinTr Configs #####################################################
# FOLDER_NAME=AdaPoinTr
# CONFIG_NAME=ScaphoidAdaPoinTr_Max.yaml

# PRETRAINED_WEIGHTS=${SCRIPT_DIR}/../pretrained_weights/AdaPoinTr_ps34.pth
############################################################################################


CONFIG=Scaphoid_models/${FOLDER_NAME}/${CONFIG_NAME}


# PRETRAINED_WEIGHTS=${SCRIPT_DIR}/../pretrained_weights/pointattn_c3d.pth
PRETRAINED_WEIGHTS=${SCRIPT_DIR}/../pretrained_weights/pointattn_pcn.pth

bash ./scripts/__train.sh 0 \
    --config_folder ${CONFIG_FOLDER} \
    --config ${CONFIG} \
    --exp_name FBaseMinDorsalPre0 \
    "$@"
    # --pretrained ${PRETRAINED_WEIGHTS} \
    
    # --resume \
    
    # --exp_name DoubleFEArchScaphoidPointAttN \
    # --resume \
    # --pretrained ${PRETRAINED_WEIGHTS} \