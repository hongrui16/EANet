#!/bin/bash

#SBATCH --partition=contrib-gpuq                    # need to set 'gpuq' or 'contrib-gpuq'  partition
#SBATCH --qos=gpu                           # need to select 'gpu' QOS or other relvant QOS
#SBATCH --job-name=eanet                   # vqgan, vqvae, vqdiffusion, gaussiandiffusion
#SBATCH --output=/home/rhong5/research_pro/hand_modeling_pro/EANet/zlog/%x-%N-%j.out   # Output file
#SBATCH --error=/home/rhong5/research_pro/hand_modeling_pro/EANet/zlog/%x-%N-%j.err    # Error file
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3                # number of cores needed
#SBATCH --gres=gpu:A100.80gb:3                # up to 8; only request what you need # gpu:A100.80gb:1 ; gpu:3g.40gb:1 
#SBATCH --mem=60gb                # 100gb for InterHand26M, memory per CORE; total memory is 1 TB (1,000,000 MB), 
#SBATCH --export=ALL 
#SBATCH --time=4-24:00:00                   # set to 24hr; please choose carefully
#SBATCH --cpus-per-task=4  # Number of CPU cores per task

set echo
umask 0027

# to see ID and state of GPUs assigned
# nvidia-smi

module load gnu10                           
module load python

source /home/rhong5/py310Torch/bin/activate
# source /home/rhong5/env_mmcv/bin/activate
cd /home/rhong5/research_pro/hand_modeling_pro/EANet
python train.py --gpu 0,1,2

# cd /home/rhong5/research_pro/hand_modeling_pro/HandPoseSD/dataloader/dataset/InterHand26M/
# python InterHand26M.py


# torchrun --nnodes=${NUM_NODES} --nproc_per_node=${GPUS_PER_NODE} train_VAE.py --debug 

# torchrun --nnodes=1 --nproc_per_node=1 train_VAE.py --batch_size 10

# salloc -p contrib-gpuq -q gpu --nodes=1 --ntasks-per-node=1 --gres=gpu:3g.40gb:1 --mem=40gb -t 0-24:00:00
# salloc -p gpuq -q gpu --nodes=1 --ntasks-per-node=1 --gres=gpu:3g.40gb:1 --mem=40gb -t 0-24:00:00
# salloc -p gpuq -q gpu --nodes=1 --ntasks-per-node=1 --gres=gpu:2g.20gb:1 --mem=40gb -t 0-24:00:00
# salloc -p contrib-gpuq -q gpu --nodes=1 --ntasks-per-node=1 --gres=gpu:2g.20gb:1 --mem=40gb -t 0-24:00:00