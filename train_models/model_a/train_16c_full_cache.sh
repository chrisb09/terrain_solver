#!/bin/zsh

#SBATCH --job-name=train_16c_full_cache_model_a
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --output=model_a_train_output_d.log
#SBATCH --error=model_a_train_error_d.log


#SBATCH --time=12:00:00
#SBATCH --account=thes2181
#SBATCH --ntasks-per-node=16
##SBATCH --mem=50G
#SBATCH --partition=c23mm
#SBATCH --mem-per-cpu=5G

##SBATCH --time=01:00:00
##SBATCH --ntasks-per-node=96
###SBATCH --mem=30G
##SBATCH --partition=devel

CORE_COUNT_PER_NODE=$SLURM_CPUS_PER_TASK
THREADS_PER_WORKER=1

cd /hpcwork/ro092286/smartsim/ || exit
. ./install.sh cpu || exit

cd mini_app/train_models/model_a || exit

# 32 CPU cores: use num-threads=32 for computation, num-workers=0 for cache mode
torchrun --nnodes=1 --nproc_per_node=$CORE_COUNT_PER_NODE train.py --cache-mode window --batch-size 8192 --epochs 1000 --max-steps 10000 --num-threads $THREADS_PER_WORKER --num-workers 2 --window-steps 50 --export-field-inference  --model transformer_mlp
# --export-field-iter --field-iter-steps 18 
