# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 6666 --nproc_per_node=4 --use_env main.py \
    # --model deit_tiny_patch16_LS --batch-size 1000 --data-set IMNET --data-path /root/imagenet-1k --output_dir ./eval_vit --eval --resume ./imagenet_ckpt/vit_tiny/checkpoint.pth
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --master_port 6668 --nproc_per_node=4 --use_env main.py \
    --model vhop_tiny_patch16_LS_hf --mode 'softmax' --step_size 1 --batch-size 1000 --data-set IMNET --data-path /root/imagenet-1k --output_dir ./eval_vhop --eval --resume ./imagenet_ckpt/vhop_tiny/checkpoint.pth
