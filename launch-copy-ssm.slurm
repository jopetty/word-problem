#!/bin/bash
#SBATCH --job-name=copy-ssm
#SBATCH --open-mode=append
#SBATCH --output=outputs/%j_%x.out
#SBATCH --error=outputs/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1

singularity exec --nv --overlay overlay-15GB-500K.ext3:ro /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash -c "

source /ext3/env.sh
conda activate wp

python src/train_copy.py train_mamba \
    --d_model $dmodel \
    --k $k \
    --vocab_size 5 \
    --batch_size $batchsize \
    --weight_decay $wd \
    --n_layers $layers \
    --type $type \
    --max_val_acc 0.995 \
"
