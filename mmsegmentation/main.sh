unset DISPLAY
export CUDA_VISIBLE_DEVICES=3

#python tools/train.py configs/sebnet/sebnet_ablation51_1xb6-160k_cityscapes.py

#python tools/train.py configs/sebnet/sebnet_ablation03_1xb6-160k_cityscapes.py
#python tools/train.py configs/sebnet/sebnet_ablation04_1xb6-160k_cityscapes.py
#python tools/train.py configs/sebnet/sebnet_ablation05_1xb6-160k_cityscapes.py
python tools/train.py configs/sebnet/sebnet_ablation41_1xb6-160k_cityscapes.py
#python tools/train.py configs/sebnet/sebnet_baseline-p-d-bas-head_1xb6_cityscapes.py
#python tools/train.py configs/sebnet/sebnet_baseline-p-d-sbd-bas-head_2xb6_mapillaryv2.py
#python tools/train.py configs/sebnet/sebnet_baseline-p-d-sbd-bas-head_1xb16_mapillaryv2.py
#python tools/train.py configs/sebnet/sebnet_baseline-p-d-sbd-bas-head_2xb8_mapillaryv2.py
#python tools/train.py configs/sebnet/sebnet_baseline-p-sbd-bas-head-conditioned_2xb6_mapillaryv2.py