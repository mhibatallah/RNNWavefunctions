#!/bin/bash
#SBATCH --time=4-00
#SBATCH --account=def-carrasqu
#SBATCH --job-name=TIM3l_3
#SBATCH --output=Outputs/TIM2D_N12x12_samp500_units100_3l_GRU_lr1e3_adap_test3.out
#SBATCH --gres=gpu:1  
#SBATCH --mem=10G               # Request the full memory of the node
#SBATCH --array=0-2   # $SLURM_ARRAY_TASK_ID takes the listed values

module load python/3.5 
module load scipy-stack
source ../tensorflow/bin/activate

python RNN_wave_function_2DTIM_GRU_lradap_test3.py $SLURM_ARRAY_TASK_ID
