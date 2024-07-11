#!/bin/bash

#SBATCH --constraint=A100
#SBATCH --time=0-04
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --exclude=compute-0-7

echo ">>> start"
source /share/nas/inigovs/venvs/rasti_byol/bin/activate
echo ">>> training"
wandb online
python finetuning.py >& finetune.log
