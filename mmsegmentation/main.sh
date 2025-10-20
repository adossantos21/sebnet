unset DISPLAY
export CUDA_VISIBLE_DEVICES=1

python tools/train.py configs/sebnet/sebnet_ablation55_1xb6-160k_cityscapes.py
