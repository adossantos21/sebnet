unset DISPLAY
export CUDA_VISIBLE_DEVICES=0,1
export PORT=29500

./tools/dist_train.sh \
    configs/sebnet/sebnet_baseline-p-d-sbd-bas-head_2xb6_mapillaryv2.py \
    2
#configs/sebnet/sebnet_baseline-p-d-bas-head_1xb6_cityscapes.py
#configs/sebnet/sebnet_baseline-p-sbd-bas-head-conditioned_2xb6_mapillaryv2.py