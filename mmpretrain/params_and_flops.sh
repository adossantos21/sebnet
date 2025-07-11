export CUDA_VISIBLE_DEVICES=0
python tools/analysis_tools/get_flops.py \
    configs/alex_sebnet/pretrain01_tests/pretrain01_staged_4xb32_in1k.py \
    --shape 224 224