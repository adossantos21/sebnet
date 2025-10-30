unset DISPLAY
export CUDA_VISIBLE_DEVICES=1

python tools/train.py configs/pidnet/pidnet-m_1xb6-241k_1024x1024-cityscapes.py
