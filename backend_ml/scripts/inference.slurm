#!/bin/bash
#SBATCH -A ECS24003
#SBATCH --time=00:59:00
#SBATCH -o log/long_rag_3.1_8b-%J.o
#SBATCH -e log/long_rag_3.1_8b-%J.e
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu-a100-small
#SBATCH --mail-user=sli@tacc.utexas.edu
#SBATCH --mail-type=all

module load tacc-apptainer

cd /scratch/07980/sli4/containers
rm taccgptback_latest.sif
apptainer pull docker://skyeglitch/taccgptback

# cd /work/07980/sli4/ls6/code/taccGPT/backend_ml
# apptainer shell --nv /scratch/07980/sli4/containers/taccgptback_latest.sif
# apptainer exec --nv /scratch/07980/sli4/containers/taccgptback_latest.sif \
# python3 rag+in_context.py \
# --path="/work/07980/sli4/ls6/code/DeepSpeedChat/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/output_Llama-3.1-8B-Instruct_24ds_ascii/" \
# --MODEL_NAME="/work/07980/sli4/ls6/code/DeepSpeedChat/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/output_Llama-3.1-8B-Instruct_24ds_ascii/"

# echo "original model output"
# apptainer exec --nv /scratch/07980/sli4/containers/taccgptback_latest.sif \
# python3 rag+in_context.py \
# --path="/scratch/07980/sli4/data/gpt/meta-llama/Meta-Llama-3.1-8B-Instruct/" \
# --MODEL_NAME="/scratch/07980/sli4/data/gpt/meta-llama/Meta-Llama-3.1-8B-Instruct/"
# echo "*************************************************************************"

# for d in /scratch/07980/sli4/data/gpt/output_Llama-3.1-8B-Instruct_ds_ascii/hf_*/ ; do
#     echo "$d"
#     echo apptainer exec --nv /scratch/07980/sli4/containers/taccgptback_latest.sif \
#     python3 rag+in_context.py \
#     --path="$d" \
#     --MODEL_NAME="$d"
#     apptainer exec --nv /scratch/07980/sli4/containers/taccgptback_latest.sif \
#     python3 rag+in_context.py \
#     --path="$d" \
#     --MODEL_NAME="$d"
#     echo "*************************************************************************"
# done

# for d in /scratch/07980/sli4/data/gpt/output_Llama-3.1-8B-Instruct_ds_utf8/hf_*/ ; do
#     echo "$d"
#     echo apptainer exec --nv /scratch/07980/sli4/containers/taccgptback_latest.sif \
#     python3 rag+in_context.py \
#     --path="$d" \
#     --MODEL_NAME="$d"
#     apptainer exec --nv /scratch/07980/sli4/containers/taccgptback_latest.sif \
#     python3 rag+in_context.py \
#     --path="$d" \
#     --MODEL_NAME="$d"
#     echo "*************************************************************************"
# done

# for d in /scratch/07980/sli4/data/gpt/output_Llama-3.1-8B-Instruct_epoch4_train_batch_size6_seq1024_lora0_zero3_20241209-093305/hf_* ; do
#     echo "$d"
#     #cp /scratch/07980/sli4/data/gpt/meta-llama/Meta-Llama-3.1-8B-Instruct/generation_config.json  $d/generation_config.json 
#     echo apptainer exec --nv /scratch/07980/sli4/containers/taccgptback_latest.sif \
#     python3 rag+in_context.py \
#     --path="$d" \
#     --MODEL_NAME="$d"
#     apptainer exec --nv /scratch/07980/sli4/containers/taccgptback_latest.sif \
#     python3 rag+in_context.py \
#     --path="$d" \
#     --MODEL_NAME="$d"
#     echo "*************************************************************************"
# done

# d="/scratch/07980/sli4/data/gpt/output_Llama-3.1-8B-Instruct_epoch4_train_batch_size6_seq1024_lora0_zero3_20241209-093305/hf_step_1000"
# apptainer exec --nv /scratch/07980/sli4/containers/taccgptback_latest.sif \
# python3 rag+in_context.py \
# --path="$d" \
# --MODEL_NAME="$d"