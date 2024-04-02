CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch  --master_port 6664 --nproc_per_node=4 --use_env memory_tune.py --model deit_tiny_patch16_LS --batch-size 256 \
     --data-path ./cifar100 --output_dir ./vit_tune_memory_tiny --step_size 1 --mode softmax --finetune ~/vision-hopfield/vit_tiny.pth
