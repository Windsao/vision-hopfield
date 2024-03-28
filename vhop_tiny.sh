# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch  --master_port 6665 --nproc_per_node=4 --use_env main.py --model vhop_tiny_patch16_LS_hf --batch-size 256 \
    # --data-path ./cifar100 --output_dir ./vhop_tiny_sft1 --step_size 1 --mode softmax1

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch  --master_port 6667 --nproc_per_node=4 --use_env main.py --model vhop_tiny_patch16_LS_hf --batch-size 256 \
    --data-path ./cifar100 --output_dir ./vhop_tiny_sp --step_size 1 --mode sparsemax

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch  --master_port 6667 --nproc_per_node=4 --use_env main.py --model vhop_tiny_patch16_LS_hf --batch-size 256 \
    # --data-path ./cifar100 --output_dir ./vhop_tiny_ent_kernel --step_size 1 --mode entmax --finetune ./imagenet_ckpt/vhop_tiny/checkpoint.pth --kernel
