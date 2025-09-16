# On the following line, `cityscapes-raw-eval` was formerly `python scripts/cityscapes_raw.py`
cityscapes-raw-eval \
/home/robert.breslin/datasets/cityscapes \
/home/robert.breslin/alessandro/paper_2/mmsegmentation/SBD/ablation07/preds/sbd_0 \
--output-path '/home/robert.breslin/alessandro/paper_2/mmsegmentation/SBD/ablation07/sbd_results' \
--nonIS \
--pre-seal \
--remove-root \
--multi-label \
--max-dist 0.02 \
--thresholds 99 \
--nproc 8 \
--split 'val' \
--pred-suffix '_SBD.png' \
#--categories '1' \ # Uncomment to evaluate on a specific category

