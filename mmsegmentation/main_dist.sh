unset DISPLAY
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PORT=29505
export GPUS=4
./tools/dist_train.sh \
    configs/sebnet/sebnet_ablation49_1xb6-160k_mapillary.py \
    $GPUS