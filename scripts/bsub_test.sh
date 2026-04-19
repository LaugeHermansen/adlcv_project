#!/bin/sh
### -- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J detr_test
### -- ask for number of cores --
#BSUB -n 1
### -- specify that the cores must be on the same host --
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -M 4GB
### -- walltime limit --
#BSUB -W 00:10
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

python -c "
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
import transformers; print('transformers:', transformers.__version__)
import scipy; print('scipy:', scipy.__version__)
print('all packages ok')
"
