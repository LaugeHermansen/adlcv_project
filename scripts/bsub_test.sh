#!/bin/sh
### -- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J detr_placement
### -- ask for number of cores --
#BSUB -n 1
### -- specify that the cores must be on the same host --
#BSUB -R "span[hosts=1]"
### -- 10GB of memory per core --
#BSUB -R "rusage[mem=1GB]"
#BSUB -M 1GB
### -- walltime limit --
#BSUB -W 00:05
### -- email --
#BSUB -u s215160@dtu.dk
### -- send notification at start and completion --
#BSUB -B
#BSUB -N
### -- output and error files --
#BSUB -o /zhome/06/9/168972/Adv_DL_CV/adlcv_project/output/out_%J.out
#BSUB -e /zhome/06/9/168972/Adv_DL_CV/adlcv_project/error/err_%J.err

mkdir -p /dtu/blackhole/10/169104/logs

module load python3/3.11.11
module load cuda/12.1

cd /zhome/06/9/168972/Adv_DL_CV/adlcv_project

# Create a real data dir we own, then symlink only Places365 (read-only is fine)
# HF cache stays inside our own data dir so we can write lock files
mkdir -p /zhome/06/9/168972/Adv_DL_CV/adlcv_project/data
ln -sfn /dtu/blackhole/10/169104/data/adlcv/Places365_trimmed \
        /zhome/06/9/168972/Adv_DL_CV/adlcv_project/data/Places365_trimmed

# Copy HF cache if not already there (2.3 GB, only runs once)
if [ ! -d /zhome/06/9/168972/Adv_DL_CV/adlcv_project/data/marco-schouten___hidden-objects ]; then
    cp -r /dtu/blackhole/10/169104/data/adlcv/marco-schouten___hidden-objects \
          /zhome/06/9/168972/Adv_DL_CV/adlcv_project/data/
fi

source /zhome/06/9/168972/.venvs/fpADLCV/bin/activate

PYTHONPATH=. python src/test_detr_pretrained.py
