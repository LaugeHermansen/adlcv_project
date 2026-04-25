#!/bin/sh
#BSUB -J adlcv_gpu_job
#BSUB -q gpuv100
#BSUB -W 24:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 8GB
#BSUB -gpu "num=1"
#BSUB -o bsub_outputs/output_%J.out
#BSUB -e bsub_outputs/error_%J.err
#BSUB -n 4
#BSUB -R "span[hosts=1]"

. "/zhome/56/e/155505/miniconda3/etc/profile.d/conda.sh"
conda activate adlcv

### ---- Run job ----
cd /zhome/56/e/155505/scratch/adlcv_project
export PYTHONPATH=/zhome/56/e/155505/scratch/adlcv_project
python src/models/diffusion.py