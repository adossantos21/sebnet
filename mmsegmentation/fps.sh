export CUDA_VISIBLE_DEVICES=2
python tools/analysis_tools/benchmark.py \
    configs/sebnet/sebnet_ablation51_1xb6-160k_cityscapes.py \
    /home/robert.breslin/alessandro/paper_2/mmpretrain/work_dirs/pidnet-imagenet/20250702_154411/checkpoints/pidnet-imagenet/20250702_154411/epoch_100.pth \
    --repeat 200