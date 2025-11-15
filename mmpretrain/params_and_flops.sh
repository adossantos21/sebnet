export CUDA_VISIBLE_DEVICES=0
python tools/analysis_tools/get_flops.py \
    configs/sebnet/sebnet_1xb64_in1k.py \
    --shape 224 224