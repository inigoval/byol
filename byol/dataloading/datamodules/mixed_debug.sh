#!/bin/bash
#SBATCH --job-name=mx-debug                    # Job name
#SBATCH --output=mx-debug_%A.log 
#SBATCH --mem=0                                     # "reserve all the available memory on each node assigned to the job"
#SBATCH --no-requeue                                    # Do not resubmit a failed job
#SBATCH --time=23:00:00                                # Time limit hrs:min:sec
#SBATCH --constraint=A100 
#SBATCH --exclusive   # only one task per node
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=24
pwd; hostname; date

nvidia-smi

PYTHON=/share/nas2/walml/miniconda3/envs/zoobot/bin/python

$PYTHON /share/nas2/walml/repos/byol/byol_main/dataloading/datamodules/mixed.py
