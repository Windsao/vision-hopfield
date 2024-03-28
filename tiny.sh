CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 6666 --nproc_per_node=4 --use_env main.py \
    --model deit_tiny_patch16_LS --batch-size 256 --data-path ./cifar100 --output_dir ./vit_tiny_ft --finetune ./imagenet_ckpt/vit_tiny/checkpoint.pth
