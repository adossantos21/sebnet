export CUDA_VISIBLE_DEVICES=1
python tools/train.py configs/sebnet/sebnet_baseline-head_1xb6_cityscapes_ce.py

#export CUDA_VISIBLE_DEVICES=1
#python tools/train.py configs/sebnet/sebnet_baseline-head_1xb8_cityscapes_ce.py

#export CUDA_VISIBLE_DEVICES=2
#python tools/train.py configs/sebnet/sebnet_baseline-head_1xb8_cityscapes_ohem.py

#export CUDA_VISIBLE_DEVICES=3
#python tools/train.py configs/sebnet/sebnet_baseline-head_1xb8_cityscapes_ce_scratch.py