unset DISPLAY
export CUDA_VISIBLE_DEVICES=0

python tools/train.py configs/pidnet/pidnet-s_1xb24-87k_512x512-bdd100k.py
