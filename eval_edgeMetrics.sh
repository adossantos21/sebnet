# On the following line, `cityscapes-raw-eval` was formerly `python scripts/cityscapes_raw.py`
cityscapes-raw-eval \
/home/robert.breslin/datasets/cityscapes \
/home/robert.breslin/alessandro/paper_2/mmsegmentation/work_dirs/sebnet_baseline-d-multilabel-head_1xb6_cityscapes/20250826_134843/SBD_preds/d_module \
--output-path '/home/robert.breslin/alessandro/paper_2/mmsegmentation/work_dirs/sebnet_baseline-d-multilabel-head_1xb6_cityscapes/20250826_134843/SBD_Results' \
--nonIS \
--pre-seal \
--remove-root \
--multi-label \
--max-dist 0.02 \
--thresholds 99 \
--nproc 8 \
--split 'val' \
--pred-suffix '.png' \
#--categories '1' \ # Uncomment to evaluate on a specific category

