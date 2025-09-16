cityscapes-raw-eval \
<path_to_cityscapes_root> \
<path_to_hed_predictions> \
--output-path <output_path> \
--categories '[1]' \
--nonIS \
--pre-seal \
--remove-root \
--max-dist 0.02 \
--thresholds 99 \
--nproc 24 \
--split 'val' \
--pred-suffix '_SBD.png' \