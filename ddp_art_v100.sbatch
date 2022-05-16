#!/bin/bash
#
#SBATCH --nodes=1                        
#SBATCH --ntasks-per-node=1              
#SBATCH --cpus-per-task=16               
#SBATCH --time=04:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=ddp_art
#SBATCH --output=ddp_art_v100.out
#SBATCH --gres=gpu:v100:2 
       

module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
conda activate test
python ddp.py --ex ddp_art_v100