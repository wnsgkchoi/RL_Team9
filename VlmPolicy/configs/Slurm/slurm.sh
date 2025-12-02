#!/bin/bash
#SBATCH --job-name=idefics_esann
#SBATCH -p boost_usr_prod
##SBATCH --qos=boost_qos_dbg
#SBATCH --qos=boost_usr_prod
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=<PATH_TO_SLURM_LOGS>/%N-%x-%j.out
#SBATCH --error=<PATH_TO_SLURM_LOGS>/%N-%x-%j.err
#SBATCH --account=<YOUR_ACCOUNT>

module load cuda/12.1
cd $FAST
echo $CONDA_DEFAULT_ENV

cd $FAST/rocket/src

export WANDB_MODE=offline
export TF_ENABLE_ONEDNN_OPTS=0
export HF_HOME=$FAST/hf_cache
export HF_DATASETS_OFFLINE=1 
export TRANSFORMERS_OFFLINE=1
echo "running with wandb set to ${WANDB_MODE}"
echo "running process ${SLURM_PROCID} on node $(hostname)"

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR


current_date=${SLURM_JOBID}_$(date +'%Y-%m-%d_%H-%M-%S')
echo $current_date

# Check if a seed is passed as first argument
if [ $# -eq 0 ]; then
    # No argument provided, set default seed
    seed=9812
else
    # Argument provided, use it as seed
    seed=$1
fi

srun accelerate launch --config_file <PATH_TO>/multi_gpu_accelerate_config.yaml ppo_crafter_parallel.py --config-path=<PATH_TO_THE_CONFIG_FOLDER> --config-name=<CONFIG_NAME> wandb_log_dir=$current_date seed=$seed track=false