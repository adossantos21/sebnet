export CUDA_VISIBLE_DEVICES=2
python tools/analysis_tools/benchmark.py \
    configs/sebnet/sebnet_baseline-casenet-head_1xb8_cityscapes.py \
    /home/robert.breslin/alessandro/paper_2/mmpretrain/work_dirs/pretrain_sebnext_large_1xb64_in1k/20250730_094937/checkpoints/pretrain_sebnext_large_1xb64_in1k/20250730_094937/epoch_300.pth \
    --repeat 200