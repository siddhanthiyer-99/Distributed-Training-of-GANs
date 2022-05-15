#!/bin/bash
#
#SBATCH --nodes=1                        
#SBATCH --ntasks-per-node=1              
#SBATCH --cpus-per-task=16               
#SBATCH --time=04:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=rtx
#SBATCH --output=rtx.out
#SBATCH --gres=gpu:rtx8000:1 
       

module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
conda activate test
python dcgan.py --device rtx8000