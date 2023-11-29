#!/bin/bash

singularity exec --overlay /scratch/jp6664/word-problem/overlay-15GB-500K.ext3:ro /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash