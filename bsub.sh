#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q gpuv100
### -- set the job Name -- 
#BSUB -J HeatmapModel
### -- ask for number of cores (default: 1) -- 
#BSUB -n 4
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 4GB of memory per core/slot -- 
#BSUB -R "rusage[mem=10GB]"
### -- specify that we want the job to get killed if it exceeds 5 GB per core/slot -- 
#BSUB -M 11GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 12:00 
### -- set the email address -- 
#BSUB -u s215225@dtu.dk
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o /work3/s215225/Output_%J.out
#BSUB -e /work3/s215225/Output_%J.err

cd /zhome/69/0/168594/Documents/advDLCV/adlcv_project/
source /zhome/69/0/168594/Documents/advDLCV/venv/bin/activate
python src/Model1.py --mode train --batch-size 256 --num-epochs 15 --freeze-text-encoder --freeze-vision-encoder --save-model-path model1.pt
