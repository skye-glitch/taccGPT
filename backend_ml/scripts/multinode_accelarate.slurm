#!/bin/bash
#SBATCH -A ECS24003
#SBATCH --time=00:59:00
#SBATCH -o log/multinotde-acc-%J.o
#SBATCH -e log/multinotde-acc-%J.e
#SBATCH -N 2
#SBATCH -n 2
#SBATCH -p gpu-a100-dev
#SBATCH --mail-user=sli@tacc.utexas.edu
#SBATCH --mail-type=all

module load tacc-apptainer

rm -r /work/07980/sli4/ls6/code/taccGPT/backend_ml/db_ce


export GPUS_PER_NODE=3
export SLURM_NNODES=2

NODEFILE=/tmp/nodelist
scontrol show hostnames $SLURM_NODELIST > $NODEFILE
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
RANKS=$(tr '\n' ' ' < $NODEFILE)

export CONTAINER="apptainer exec --nv /scratch/07980/sli4/containers/pipeline_latest.sif "
export LAUNCHER="accelerate launch \
    --multi_gpu \
    --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
    --num_machines $SLURM_NNODES \
    --main_process_ip $head_node_ip \
    --main_process_port 29500 \
    --rdzv_backend c10d \
    "
export SCRIPT="/work/07980/sli4/ls6/code/taccGPT/backend_ml/accelarate_ce.py"
export ARGS="--model_id=\"meta-llama/Meta-Llama-3.1-8B-Instruct\""

EXP="export HF_HOME="/tmp/huggingface_cache" ";
EXP+="export HF_HUB_CACHE="$SCRATCH/huggingface_cache" ";
$EXP
EXP2="export ACCELERATE_DIR="$HF_HOME/accelerate"  ";
$EXP


# This step is necessary because accelerate launch does not handle multiline arguments properly
export CMD="$CONTAINER $LAUNCHER $SCRIPT $ARGS" 



echo $RANKS
RANK=0
for NODE in $RANKS; do
    CMD="$CONTAINER accelerate launch --multi_gpu --num_processes=6 --num_machines=2 --machine_rank=$RANK --main_process_ip $head_node_ip --main_process_port 29500 --rdzv_backend c10d $SCRIPT $ARGS"
    if [[ $NODE == $head_node_ip ]]; then
        echo $CMD
	    $CMD 
    else
        echo "Launching rank $RANK on remote node $NODE"
	    ssh $NODE "cd $PWD;bash -lc 'module load tacc-apptainer; export OMP_NUM_THREADS=8; $EXP $EXP2 $CMD'" 
    fi
    RANK=$((RANK + 1))
done