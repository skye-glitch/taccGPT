#!/bin/bash
#SBATCH -A ECS24003
#SBATCH --time=00:59:00
#SBATCH -o log/multinotde-torch-ce-%J.o
#SBATCH -e log/multinotde-torch-ce-%J.e
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

export CONTAINER="apptainer exec --nv /scratch/07980/sli4/containers/taccgptback_latest.sif "
export LAUNCHER="accelerate launch \
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
$EXP2

# source /work/07980/sli4/ls6/code/DeepSpeedChat/applications/DeepSpeed-Chat/venv/bin/activate
# HUGGINGFACE_TOKEN_file="/work/07980/sli4/ls6/code/DeepSpeedChat/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/huggingface_token"
# HUGGINGFACE_TOKEN=$( cat $HUGGINGFACE_TOKEN_file )
# HUGGINGFACE_CMD="huggingface-cli login --token $HUGGINGFACE_TOKEN"
# $HUGGINGFACE_CMD

 


RANK=0
for NODE in $RANKS; do
    LAUNCHER="apptainer exec --nv /scratch/07980/sli4/containers/taccgptback_latest.sif torchrun --nproc_per_node=3 --nnodes=2 --node_rank=$RANK --master_addr=$head_node_ip --master_port=1234"
    FULL_CMD="$LAUNCHER pippy_llama.py"
    if [[ $NODE == $head_node_ip ]]; then
        echo $FULL_CMD
	    eval $FULL_CMD &
    else
        echo "Launching rank $RANK on remote node $NODE"
	    ssh $NODE "cd $PWD;bash -lc 'module load tacc-apptainer; export OMP_NUM_THREADS=8; $EXP; $EXP2; $HUGGINGFACE_CMD; $FULL_CMD'" &
    fi
    RANK=$((RANK + 1))
done


apptainer exec --nv /scratch/07980/sli4/containers/taccgptback_latest.sif accelerate launch --multi_gpu --num_processes=6 --num_machines=2 --machine_rank=$RANK --main_process_ip c301-002 --main_process_port 29500 --rdzv_backend c10d /work/07980/sli4/ls6/code/taccGPT/backend_ml/accelarate_ce.py --model_id="meta-llama/Meta-Llama-3.1-8B-Instruct"


torchrun --nproc_per_node=3 --nnodes=1 --node_rank=0 --master_addr=c301-004 --master_port=1234 pippy_llama.py
  