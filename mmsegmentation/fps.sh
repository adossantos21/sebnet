export CUDA_VISIBLE_DEVICES=0
python tools/analysis_tools/benchmark.py \
    configs/pidnet/pidnet-l_2xb6-120k_1024x1024-cityscapes.py \
    /home/robert.breslin/alessandro/paper_2/mmpretrain/work_dirs/pidnet-imagenet/20250702_154411/checkpoints/pidnet-imagenet/20250702_154411/epoch_100.pth \
    --repeat 200