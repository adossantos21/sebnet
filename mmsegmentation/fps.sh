export CUDA_VISIBLE_DEVICES=0
python tools/analysis_tools/benchmark.py \
    configs/pidnet/pidnet-s_1xb24-87k_512x512-bdd100k.py \
    path/to/checkpoint.pth \
    --repeat 200