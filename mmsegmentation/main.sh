unset DISPLAY
export CUDA_VISIBLE_DEVICES=0

python tools/train.py configs/sebnet/sebnet_ablation39_1xb6-160k_cityscapes.py
