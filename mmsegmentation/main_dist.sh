#unset DISPLAY
#export CUDA_VISIBLE_DEVICES=2,3
#export PORT=29505
#export TORCH_DISTRIBUTED_DEBUG=DETAIL
#./tools/dist_train.sh \
#    configs/sebnet/sebnet_baseline-p-sbd-bas-head-conditioned_2xb6_mapillaryv2.py \
#    2

#configs/sebnet/sebnet_baseline-p-d-sbd-bas-head_2xb6_mapillaryv2.py
#configs/sebnet/sebnet_baseline-p-sbd-bas-head-conditioned_2xb6_mapillaryv2.py
#configs/sebnet/sebnet_baseline-p-d-bas-head_1xb6_cityscapes.py