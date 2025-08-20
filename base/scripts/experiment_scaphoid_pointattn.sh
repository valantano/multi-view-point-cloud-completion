#!/usr/bin/zsh

# Default job name if not provided
FOLDER_NAME=${1}
CONFIG_NAME=${2}
EXP_NAME=${3}
JOB_NAME=${4:-Exp}

STRICT=${5}

TEMP_DIR=${TMPDIR:-/home/$USER/tmp}
mkdir -p $TEMP_DIR

# Create a temporary SLURM script
TEMP_SCRIPT=$(mktemp "$TEMP_DIR/tmp.XXXXXXXXXX")

cat <<EOT > $TEMP_SCRIPT
#!/usr/bin/zsh
#SBATCH --job-name=${JOB_NAME}
#SBATCH -A rwth0536
#SBATCH --time=0-32:00              			# Runtime in D-HH:MM
#SBATCH --output=output.%J.log         			# File to which STDOUT will be written
#SBATCH --mail-type=ALL         			# Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=valentino.geuenich@rwth-aachen.de     	# Email to which notifications will be sent
#SBATCH --gres=gpu:1
#SBATCH --mem=10G

module restore mt9

source ~/.bashrc

# Start training

if [ "\$USER" = "valantano" ]; then
    SCRIPT_DIR=/home/valantano/mt/repository/base/scripts/
else
    SCRIPT_DIR='/home/fn848825/multi-view-point-cloud-completion/base/scripts/'
fi

cd "\$SCRIPT_DIR"
cd "../"

CONFIG_FOLDER=\${SCRIPT_DIR}/../cfgs/


CONFIG=Scaphoid_models/${FOLDER_NAME}/${CONFIG_NAME}


# PRETRAINED_WEIGHTS=\${SCRIPT_DIR}/../pretrained_weights/AdaPoinTr_ps34.pth
PRETRAINED_WEIGHTS=\${SCRIPT_DIR}/../pretrained_weights/pointattn_pcn.pth


bash ./scripts/__train.sh 0 \
    --config_folder \${CONFIG_FOLDER} \
    --config \${CONFIG} \
    --exp_name ${EXP_NAME} \
    --val_freq 1 \
    --strict ${STRICT:-'False'} \
    --pretrained \${PRETRAINED_WEIGHTS}
EOT

# Submit the temporary script
sbatch $TEMP_SCRIPT
# bash $TEMP_SCRIPT

# Clean up the temporary script after submission
rm $TEMP_SCRIPT

#     --pretrained \${PRETRAINED_WEIGHTS}
