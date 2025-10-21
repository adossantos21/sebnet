unset DISPLAY
export CUDA_VISIBLE_DEVICES=2

python tools/train.py configs/pidnet/pidnet-l_1xb12-6k_960x720-camvid.py
