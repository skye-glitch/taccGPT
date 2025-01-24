#!/bin/bash
#SBATCH -A ECS24003
#SBATCH --time=00:15:00
#SBATCH -o log/multinotde-torch-ce-%J.o
#SBATCH -e log/multinotde-torch-ce-%J.e
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu-a100
#SBATCH --mail-user=sli@tacc.utexas.edu
#SBATCH --mail-type=all


module load tacc-apptainer

cd /scratch/07980/sli4/containers
apptainer pull docker://skyeglitch/pipeline_latest

export GPUS_PER_NODE=3
export SLURM_NNODES=1

NODEFILE=/tmp/nodelist
scontrol show hostnames $SLURM_NODELIST > $NODEFILE
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
RANKS=$(tr '\n' ' ' < $NODEFILE)
echo $RANKS

#ARGS="--model_id=\"meta-llama/Meta-Llama-3.1-8B-Instruct\""

PRELOAD="module load tacc-apptainer  "
EXP="export HF_HOME=/tmp/huggingface_cache  "
EXP+="export HF_HUB_CACHE=$SCRATCH/huggingface_cache  "
EXP+="export ACCELERATE_DIR=$HF_HOME/accelerate  "
EXP+="export OMP_NUM_THREADS=8 "
EXP+="export TOKENIZERS_PARALLELISM=false"

$PRELOAD
$EXP

apptainer exec --nv /scratch/07980/sli4/containers/pipeline_latest.sif accelerate --nproc-per-node 3 accelerate_ce.py
