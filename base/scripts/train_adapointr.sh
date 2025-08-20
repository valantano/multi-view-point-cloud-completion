#!/usr/bin/zsh

#SBATCH --job-name=adapointr

#SBATCH --time=2-00:00              			# Runtime in D-HH:MM
#SBATCH --output=output.%J.log         			# File to which STDOUT will be written
#SBATCH --mail-type=ALL         			# Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=broessner@hia.rwth-aachen.de     	# Email to which notifications will be sent
#SBATCH --gres=gpu:2    

# Activate python environment
source ~/.bashrc
source ~/.zshrc
conda activate mt9

# Start training

# python main.py --config ./cfgs/Scar_models/AdaPoinTr.yaml --exp_name Scar_1Side_RandomRot --ckpts ./pretrained/AdaPoinTr_ps55.pth --ckpts ./experiments/AdaPoinTr/Scar_models/Scar_1Side_RandRot/ckpt-best.pth
# bash ./scripts/train.sh 0,1 \

SCRIPT_DIR=$(dirname "$(realpath "$0")")
cd "$SCRIPT_DIR"
cd "../"

CONFIG_FOLDER=${SCRIPT_DIR}/../cfgs/
CONFIG=Scaphoid_models/AdaPoinTr.yaml

PRETRAINED_WEIGHTS=${SCRIPT_DIR}/../pretrained_weights/AdaPoinTr_ps34.pth

bash ./scripts/__train.sh 0 \
    --config_folder ${CONFIG_FOLDER} \
    --config ${CONFIG} \
    --exp_name Pretrained \
    --resume \
    # --pretrained ${PRETRAINED_WEIGHTS} \

