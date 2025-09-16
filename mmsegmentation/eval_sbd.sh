cityscapes-raw-eval \
<path_to_cityscapes_root> \
<path_to_sbd_predictions> \
--output-path <output_path> \
--nonIS \
--pre-seal \
--remove-root \
--multi-label \
--categories '[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]' \
--max-dist 0.02 \
--thresholds 99 \
--nproc 8 \
--split 'val' \
--pred-suffix '_SBD.png' \