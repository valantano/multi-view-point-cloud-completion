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



FOLDER_NAME=PoseEstCfgs
CONFIG_NAME=PoseEst_ScaphoidPointAttN_dorsal.yaml
# CONFIG_NAME=PoseEst_ScaphoidPointAttN_volar.yaml
CONFIG_NAME=PoseEst_ScaphoidPointAttN_both.yaml


CONFIG=Scaphoid_models/${FOLDER_NAME}/${CONFIG_NAME}

# CONFIG=Scaphoid_models/DoubleFEArchScaphoidPointAttN.yaml


# PRETRAINED_WEIGHTS=${SCRIPT_DIR}/../pretrained_weights/pointattn_c3d.pth
PRETRAINED_WEIGHTS=${SCRIPT_DIR}/../pretrained_weights/pointattn_pcn.pth

bash ./scripts/__train_pose_est_both.sh 0 \
    --config_folder ${CONFIG_FOLDER} \
    --config ${CONFIG} \
    --exp_name +DebugPoseEstBothNewStatic \
    --strict False \
    "$@"
    # --pretrained ${PRETRAINED_WEIGHTS} \
    
    # --resume \
    
    # --exp_name DoubleFEArchScaphoidPointAttN \
    # --resume \
    # --pretrained ${PRETRAINED_WEIGHTS} \
    



