#!/usr/bin/zsh

#SBATCH --job-name=SMAC3DorsalPose
#SBATCH -A thes1928


#SBATCH --time=0-90:00              			# Runtime in D-HH:MM
#SBATCH --output=output.%J.log         			# File to which STDOUT will be written
#SBATCH --mail-type=ALL         			# Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=valentino.geuenich@rwth-aachen.de     	# Email to which notifications will be sent
#SBATCH --gres=gpu:1
#SBATCH --mem=10G

module restore mt9
conda activate mt

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



# FOLDER_NAME=StaticCfgs
# CONFIG_NAME=ScaphoidPointAttN_Baseline_Min_dorsal_static.yaml
# CONFIG_NAME=ScaphoidPointAttN_Baseline_Min_volar_static.yaml
# CONFIG_NAME=ScaphoidPointAttN_Baseline_Max_static.yaml
# CONFIG_NAME=ScaphoidPointAttN_Double_Net.yaml

# FOLDER_NAME=CompletionPoseEstCfgs
# CONFIG_NAME=CompletionPoseEst_ScaphoidPointAttN_dorsal.yaml
# # # CONFIG_NAME=CompletionPoseEst_ScaphoidPointAttN_volar.yaml
# CONFIG_NAME=CompletionPoseEst_concat_input_dorsal.yaml



#################### AdaPoinTr Configs #####################################################
# FOLDER_NAME=AdaPoinTr
# CONFIG_NAME=ScaphoidAdaPoinTr_Max.yaml

# PRETRAINED_WEIGHTS=${SCRIPT_DIR}/../pretrained_weights/AdaPoinTr_ps34.pth
############################################################################################
FOLDER_NAME=PoseEstCfgs
CONFIG_NAME=PoseEst_ScaphoidPointAttN_dorsal.yaml
PROJECT_POSTFIX=PoseEst_dorsal
CONFIG_NAME=PoseEst_ScaphoidPointAttN_volar.yaml
PROJECT_POSTFIX=PoseEst_volar
# CONFIG_NAME=PoseEst_ScaphoidPointAttN_both.yaml
# PROJECT_POSTFIX=PoseEst_both

# FOLDER_NAME=StaticCfgs
# CONFIG_NAME=ScaphoidPointAttN_Baseline_Min_dorsal_static.yaml
# PROJECT_POSTFIX=Static_dorsal
# CONFIG_NAME=ScaphoidPointAttN_Baseline_Min_volar_static.yaml
# PROJECT_POSTFIX=Static_volar
# CONFIG_NAME=ScaphoidPointAttN_Baseline_Max_static.yaml
# PROJECT_POSTFIX=Static_max


CONFIG=Scaphoid_models/${FOLDER_NAME}/${CONFIG_NAME}


# PRETRAINED_WEIGHTS=${SCRIPT_DIR}/../pretrained_weights/pointattn_c3d.pth
PRETRAINED_WEIGHTS=${SCRIPT_DIR}/../pretrained_weights/pointattn_pcn.pth

bash ./scripts/__smac3.sh 0 \
    --config_folder ${CONFIG_FOLDER} \
    --config_file ${CONFIG} \
    --project_postfix ${PROJECT_POSTFIX} \
    "$@"
    # --pretrained ${PRETRAINED_WEIGHTS} \
    
    # --resume \
    
    # --exp_name DoubleFEArchScaphoidPointAttN \
    # --resume \
    # --pretrained ${PRETRAINED_WEIGHTS} \