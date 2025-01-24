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




export GPUS_PER_NODE=3
export SLURM_NNODES=2

NODEFILE=/tmp/nodelist
scontrol show hostnames $SLURM_NODELIST > $NODEFILE
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
RANKS=$(tr '\n' ' ' < $NODEFILE)


SCRIPT="/work/07980/sli4/ls6/code/taccGPT/backend_ml/accelarate_ce.py"
ARGS="--model_id=\"meta-llama/Meta-Llama-3.1-8B-Instruct\""

EXP="export HF_HOME=/tmp/huggingface_cache  "
EXP+="export HF_HUB_CACHE=$SCRATCH/huggingface_cache  "
EXP+="export ACCELERATE_DIR=$HF_HOME/accelerate  "
EXP+="export OMP_NUM_THREADS=8 "


module load tacc-apptainer
apptainer exec --nv /scratch/07980/sli4/containers/pipeline_latest.sif accelerate launch --num_processes 3 accelarate_ce.py


echo $RANKS
RANK=0
for NODE in $RANKS; do
    CMD="module load tacc-apptainer; apptainer exec --nv /scratch/07980/sli4/containers/pipeline_latest.sif accelerate launch --multi_gpu --num_processes=6 --num_machines=2 --machine_rank=$RANK --main_process_ip $head_node_ip --main_process_port 29500 --rdzv_backend c10d $SCRIPT "
    if [[ $NODE == $head_node_ip ]]; then
        echo $CMD
	    eval $CMD &
    else
        echo "Launching rank $RANK on remote node $NODE"
        echo $CMD
	    ssh $NODE "cd $PWD;module load tacc-apptainer; $EXP; $CMD" &
    fi
    RANK=$((RANK + 1))
done