#!/bin/bash
#SBATCH -A p32013 ## Required: your allocation/account name, i.e. eXXXX, pXXXX or bXXXX
#SBATCH -p gengpu ## Required: (buyin, short, normal, long, gengpu, genhimem, etc)
#SBATCH --gres gpu:a100:4
#SBATCH --constraint=sxm

#SBATCH -t 48:00:00 ## Required: How long will the job need to run (remember different partitions have restrictions on this parameter)
#SBATCH --nodes 1 ## how many computers/nodes do you need (no default)
#SBATCH --cpus-per-task 24
#SBATCH --ntasks-per-node 2 ## how many cpus or processors do you need on per computer/node (default value 1)
#SBATCH --mem 240G ## how much RAM do you need per computer/node (this affects your FairShare score so be careful to not ask for more than you need))
#SBATCH --job-name=deit_tiny ## When you run squeue -u 
#SBATCH --mail-type END ## BEGIN, END, FAIL or ALL
#SBATCH --error=robin/deit_tiny.err
#SBATCH --output=robin/deit_tiny.out

module purge
module load python-miniconda3/4.12.0
module load moose/1.0.0
module load cuda/11.4.0-gcc
module load gcc/9.2.0

conda init bash
source ~/.bashrc
# conda create -n maehop python=3.9

conda activate maehop

cd /projects/p32013/Shang/deit

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR


/home/hlv8980/.conda/envs/maehop/bin/python3.9 -m torch.distributed.run --nproc_per_node=4 --use_env main.py \
  --model deit_tiny_patch16_LS --batch-size 256 --output_dir ./vit_tiny --data-path /projects/p32013/Hopfield_archieved/vit-pytorch/data/imagenet-1k 
