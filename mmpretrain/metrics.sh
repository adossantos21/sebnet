export CUDA_VISIBLE_DEVICES=0
python tools/test.py \
    configs/pidnet/pidnet-imagenet.py \
    work_dirs/pidnet_L_imagenet_ckpt/backbone_pidnet-l_imagenet1k_20230306-67889109.pth \
    --out "work_dirs/pidnet-imagenet/metrics.txt" \
    --out-item metrics