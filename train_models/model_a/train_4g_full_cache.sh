#!/bin/zsh

#SBATCH --job-name=train_4g_full_cache_model_a
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=5G
#SBATCH --output=model_a_train_output_gpu.log
#SBATCH --error=model_a_train_error_gpu.log
#SBATCH --partition=c23g
#SBATCH --account=thes2181
#SBATCH --gpus-per-task=2

cd /hpcwork/ro092286/smartsim/ || exit
source ./install.sh cuda-12 || exit

cd mini_app/train_models/model_a || exit

# 2 GPUs (H100), full cache, larger batch
torchrun --nproc_per_node=2 train.py --cache-mode window --batch-size 131072 --epochs 1000 --max-steps 10000 --num-threads 1 --num-workers 0 --window-steps 50 --export-field-inference --model transformer_mlp