#!/bin/zsh

#SBATCH --job-name=train_1c_window_model_a
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem=20G
#SBATCH --output=model_a_train_output_window.log
#SBATCH --error=model_a_train_error_window.log
#SBATCH --partition=devel

cd /hpcwork/ro092286/smartsim/ || exit
source ./install.sh || exit

cd mini_app/train_models/model_a || exit

# Single process with 4 CPU cores, window cache mode
python train.py --cache-mode window --window-steps=20 --batch-size 1000 --epochs 5 --max-steps 1000 --num-threads 4 --num-workers 2