#!/bin/bash
#
#SBATCH --nodes=1                        
#SBATCH --ntasks-per-node=1              
#SBATCH --cpus-per-task=16               
#SBATCH --time=05:00:00
#SBATCH --mem=4GB
#SBATCH --job-name=ff_dp
#SBATCH --output=ff_dp.out
#SBATCH --gres=gpu:rtx8000:1 
       

module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
conda activate test
python dataparallelgan.py --ex ff_dataparallel