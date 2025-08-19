export CUDA_VISIBLE_DEVICES=2
python tools/analysis_tools/get_flops.py \
    configs/sebnet/test_sebnextalt_staged_4xb32_in1k.py \
    --shape 2048 1024