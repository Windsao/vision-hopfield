CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --master_port 6667 --nproc_per_node=4 --use_env main.py --model deit_base_patch16_LS --batch-size 32 --data-path ./cifar100 --output_dir ./vit_base
