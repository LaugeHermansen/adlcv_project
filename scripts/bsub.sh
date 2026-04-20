#!/bin/sh
### -- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J detr_placement
### -- ask for number of cores --
#BSUB -n 4
### -- specify that the cores must be on the same host --
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -M 16GB
### -- walltime limit --
#BSUB -W 04:00
### -- email --
#BSUB -u s215160@dtu.dk
#BSUB -B
#BSUB -N
### -- output and error files --
#BSUB -o /zhome/06/9/168972/Adv_DL_CV/adlcv_project/output/out_%J.out
#BSUB -e /zhome/06/9/168972/Adv_DL_CV/adlcv_project/error/err_%J.err

module load python3/3.11.11
module load cuda/12.1

cd /zhome/06/9/168972/Adv_DL_CV/adlcv_project
source /zhome/06/9/168972/.venvs/fpADLCV/bin/activate

WANDB_MODE=online PYTHONPATH=. python src/test_detr_pretrained.py
