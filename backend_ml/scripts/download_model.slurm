#!/bin/bash
#SBATCH -A ECS24003
#SBATCH --time=05:59:00
#SBATCH -o log/download-%J.o
#SBATCH -e log/download-%J.e
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p vm-small
#SBATCH --mail-user=sli@tacc.utexas.edu
#SBATCH --mail-type=all

# module load cuda/12.0
# export CUDA_HOME=/opt/apps/cuda/12.0
# export NCCL_HOME=/opt/apps/cuda12_0/nccl/2.17.1/
# export CUDNN_HOME=/opt/apps/cuda12_0/cudnn/8.8.1 
export LD_LIBRARY_PATH=/usr/lib64:/opt/apps/cuda/12.0/lib64:/opt/intel/compilers_and_libraries_2020.1.217/linux/bin/intel64:/opt/apps/intel19/python3/3.9.7/lib:/opt/apps/pmix/3.2.3/lib:/opt/intel/compilers_and_libraries_2020.4.304/linux/mpi/intel64/libfabric/lib:/opt/intel/compilers_and_libraries_2020.4.304/linux/mpi/intel64/lib/release:/opt/intel/compilers_and_libraries_2020.4.304/linux/mpi/intel64/lib:/opt/intel/debugger_2020/libipt/intel64/lib:/opt/intel/compilers_and_libraries_2020.1.217/linux/daal/lib/intel64_lin:/opt/intel/compilers_and_libraries_2020.1.217/linux/tbb/lib/intel64_lin/gcc4.8:/opt/intel/compilers_and_libraries_2020.1.217/linux/mkl/lib/intel64_lin:/opt/intel/compilers_and_libraries_2020.1.217/linux/ipp/lib/intel64:/opt/intel/compilers_and_libraries_2020.1.217/linux/compiler/lib/intel64_lin:/opt/apps/gcc/9.4.0/lib64:/opt/apps/gcc/9.4.0/lib

source /work/07980/sli4/ls6/code/DeepSpeedChat/applications/DeepSpeed-Chat/venv/bin/activate

export PATH=/work/07980/sli4/ls6/code/DeepSpeedChat/applications/DeepSpeed-Chat/training/pdsh/bin:/:$PATH
export HF_HOME=/tmp/huggingface_cache
HUGGINGFACE_TOKEN_file="/work/07980/sli4/ls6/code/DeepSpeedChat/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/huggingface_token"
HUGGINGFACE_TOKEN=$( cat $HUGGINGFACE_TOKEN_file )
srun huggingface-cli login --token $HUGGINGFACE_TOKEN

huggingface-cli download meta-llama/Meta-Llama-3.1-405B-Instruct  --local-dir /scratch/07980/sli4/data/gpt/meta-llama/Meta-Llama-3.1-405B-Instruct


