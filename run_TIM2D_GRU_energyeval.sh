#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --account=def-carrasqu
#SBATCH --job-name=TIM2D_GRU
#SBATCH --output=Outputs/TIM2D_N12x12_units100_3l_1000samp_GRU_lr1e3_adap_energyeval.out
#SBATCH --gres=gpu:1             # Number of GPUs (per node)
#SBATCH --mem=64G               # memory (per node)

module load python/3.5 
module load scipy-stack
source ../tensorflow/bin/activate

python RNN_wave_function_2DTIM_GRU_energyeval.py