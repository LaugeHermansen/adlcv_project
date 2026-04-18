#!/bin/sh
### -- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J detr_placement
### -- ask for number of cores --
#BSUB -n 4
### -- specify that the cores must be on the same host --
#BSUB -R "span[hosts=1]"
### -- 10GB of memory per core --
#BSUB -R "rusage[mem=10GB]"
#BSUB -M 11GB
### -- walltime limit --
#BSUB -W 12:00
### -- email --
#BSUB -u s214743@dtu.dk
### -- send notification at start and completion --
#BSUB -B
#BSUB -N
### -- output and error files --
#BSUB -o /dtu/blackhole/10/169104/logs/detr_%J.out
#BSUB -e /dtu/blackhole/10/169104/logs/detr_%J.err

mkdir -p /dtu/blackhole/10/169104/logs

module load python3/3.11.11
module load cuda/12.1

cd /zhome/06/9/168972/Adv_DL_CV/adlcv_project

# Point ./data at the blackhole data directory
ln -sfn /dtu/blackhole/10/169104/data/adlcv ./data

source /zhome/06/9/168972/.venvs/fpADLCV/bin/activate

PYTHONPATH=. python src/test_detr_pretrained.py
