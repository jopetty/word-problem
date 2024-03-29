#!/bin/bash
#SBATCH --job-name=word-problem-ssm
#SBATCH --open-mode=append
#SBATCH --output=outputs/%j_%x.out
#SBATCH --error=outputs/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1

singularity exec --nv --overlay overlay-15GB-500K.ext3:ro /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash -c "

source /ext3/env.sh
conda activate wp

python src/main.py train_mamba \
    --group $group \
    --k $k \
    --batch_size $batchsize \
    --weight_decay $wd \
    --n_layers $layers \
    --max_samples $samples \
    --d_model 698 \
    --strict_len \
    --weight_sharing $wsharing \
"

