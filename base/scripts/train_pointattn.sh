#!/usr/bin/zsh

#SBATCH --job-name=PointAttn

#SBATCH --time=0-04:00              			# Runtime in D-HH:MM
#SBATCH --output=output.%J.log         			# File to which STDOUT will be written
#SBATCH --mail-type=ALL         			# Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=valentino.geuenich@rwth-aachen.de     	# Email to which notifications will be sent
#SBATCH --gres=gpu:1
#SBATCH --mem=10G

module restore mt9

source ~/.bashrc



# Start training

# python main.py --config ./cfgs/Scar_models/AdaPoinTr.yaml --exp_name Scar_1Side_RandomRot --ckpts ./pretrained/AdaPoinTr_ps55.pth --ckpts ./experiments/AdaPoinTr/Scar_models/Scar_1Side_RandRot/ckpt-best.pth
# bash ./scripts/train.sh 0,1 \

# if user is not valantano then use second script_dir
if [ "$USER" = "valantano" ]; then
    SCRIPT_DIR=$(dirname "$(realpath "$0")")
else
    SCRIPT_DIR='/home/fn848825/multi-view-point-cloud-completion/base/scripts/'
fi


cd "$SCRIPT_DIR"
cd "../"

CONFIG_FOLDER=${SCRIPT_DIR}/../cfgs/
CONFIG=Scaphoid_models/PointAttN.yaml

# PRETRAINED_WEIGHTS=${SCRIPT_DIR}/../pretrained_weights/pointattn_c3d.pth
PRETRAINED_WEIGHTS=${SCRIPT_DIR}/../pretrained_weights/pointattn_pcn.pth

bash ${SCRIPT_DIR}/__train.sh 0 \
    --config_folder ${CONFIG_FOLDER} \
    --config ${CONFIG} \
    --exp_name ModularizedArch \
    # --pretrained ${PRETRAINED_WEIGHTS} \
    # --resume \
    # --pretrained ${PRETRAINED_WEIGHTS} \
    



