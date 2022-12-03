#!/bin/bash
#SBATCH -p GPU # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH -t 0-36:00 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#SBATCH --gres=gpu:1

source activate convnext
cd /home/u867803/Projects/Thesis/Scripts/convnext/2/ConvNeXt-main

python -m torch.distributed.launch --nproc_per_node=1 main.py \
--model convnext_tiny --drop_path 0.1 \
--batch_size 64 --lr 4e-3 --update_freq 32 \
--model_ema true --model_ema_eval true \
--data_path /home/u867803/Projects/Thesis/DataSets/ImageNet \
--output_dir /home/u867803/Projects/Thesis/ModelResults/20221126_ConvNext_ImageNet