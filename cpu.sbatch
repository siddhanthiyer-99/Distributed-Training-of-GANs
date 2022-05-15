#!/bin/bash
#
#SBATCH --nodes=1                        
#SBATCH --ntasks-per-node=1              
#SBATCH --cpus-per-task=16               
#SBATCH --time=04:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=cpu
#SBATCH --output=cpu.out
       

module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
conda activate test
python dcgan.py --device cpu