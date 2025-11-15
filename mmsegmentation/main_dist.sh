unset DISPLAY
export CUDA_VISIBLE_DEVICES=0,1
export PORT=29505
export GPUS=2
./tools/dist_train.sh \
    configs/pidnet/pidnet-m_2xb6-120k_1024x1024-cityscapes.py \
    $GPUS