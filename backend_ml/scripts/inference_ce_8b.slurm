#!/bin/bash
#SBATCH -A ECS24003
#SBATCH --time=3:59:00
#SBATCH -o log/ce8-%J.o
#SBATCH -e log/ce8-%J.e
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu-a100
#SBATCH --mail-user=sli@tacc.utexas.edu
#SBATCH --mail-type=all

module load tacc-apptainer

echo "4 bit quantization"
apptainer exec --nv /scratch/07980/sli4/containers/taccgptback_latest.sif \
python3 rag_ce.py \
--path="/scratch/07980/sli4/data/gpt/meta-llama/Meta-Llama-3.1-8B-Instruct/" \
--MODEL_NAME="/scratch/07980/sli4/data/gpt/meta-llama/Meta-Llama-3.1-8B-Instruct/" \
--quant="4"

# echo "8 bit quantization"
# apptainer exec --nv /scratch/07980/sli4/containers/taccgptback_latest.sif \
# python3 rag_ce.py \
# --path="/scratch/07980/sli4/data/gpt/meta-llama/Meta-Llama-3.1-8B-Instruct/" \
# --MODEL_NAME="/scratch/07980/sli4/data/gpt/meta-llama/Meta-Llama-3.1-8B-Instruct/" \
# --quant="8"

# echo "no quantization"
# apptainer exec --nv /scratch/07980/sli4/containers/taccgptback_latest.sif \
# python3 rag_ce.py \
# --path="/scratch/07980/sli4/data/gpt/meta-llama/Meta-Llama-3.1-8B-Instruct/" \
# --MODEL_NAME="/scratch/07980/sli4/data/gpt/meta-llama/Meta-Llama-3.1-8B-Instruct/" 