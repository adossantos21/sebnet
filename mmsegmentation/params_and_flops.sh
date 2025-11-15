export CUDA_VISIBLE_DEVICES=0
python tools/analysis_tools/get_flops.py \
    configs/pidnet/pidnet-l_1xb6-241k_1024x1024-cityscapes.py \
    --shape 2048 1024