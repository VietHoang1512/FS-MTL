#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --time=47:59:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=100GB
#SBATCH --job-name=city_scape
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
conda activate /scratch/hvp2011/envs/python310
nvidia-smi

cd /scratch/hvp2011/implement/FS-MTL/city2scape
python  train.py  --data-path data --model mtan --seed 0  --method cagrad --batch-size 8 --lr 0.0001 --adaptive False --rho .0
