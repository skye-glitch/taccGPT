#!/bin/bash
#SBATCH -A ECS24003
#SBATCH --time=00:15:00
#SBATCH -o log/multinotde-torch-ce-%J.o
#SBATCH -e log/multinotde-torch-ce-%J.e
#SBATCH -N 2
#SBATCH -n 2
#SBATCH -p gpu-a100
#SBATCH --mail-user=sli@tacc.utexas.edu
#SBATCH --mail-type=all


module load tacc-apptainer

export GPUS_PER_NODE=3
export SLURM_NNODES=2

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

$EXP
$PRELOAD
apptainer exec --nv /scratch/07980/sli4/containers/pipeline_2.4.sif torchrun --nproc-per-node 3 pippy_llama.py

RANK=0
for NODE in $RANKS; do
    LAUNCHER="apptainer exec --nv /scratch/07980/sli4/containers/pipeline_2.4.sif torchrun --nproc_per_node=3 --nnodes=2 --node_rank=$RANK --master_addr=$head_node_ip --master_port=1234"
    FULL_CMD="$PRELOAD; $EXP; $LAUNCHER pippy_llama.py"
    if [[ $NODE == $head_node_ip ]]; then
        echo $FULL_CMD
	    eval $FULL_CMD &
    else
        echo "Launching rank $RANK on remote node $NODE"
        echo $FULL_CMD
        echo "ssh to $NODE"
	    ssh $NODE "cd $PWD; module load tacc-apptainer; $EXP; $FULL_CMD" &
    fi
    RANK=$((RANK + 1))
done




#https://github.com/gpauloski/kfac-pytorch/blob/main/scripts/run_imagenet.sh