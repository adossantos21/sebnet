export CUDA_VISIBLE_DEVICES=0
python tools/test.py \
    configs/pidnet/pidnet-l_1xb6-241k_1024x1024-cityscapes.py \
    work_dirs/path/to/checkpoint \
    --out "work_dirs/pidnet-l_1xb6-241k_1024x1024-cityscapes/metrics.txt" \
    --out-item metrics