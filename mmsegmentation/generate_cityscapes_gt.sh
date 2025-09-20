# Use `-h` flag to view all optional arguments and corresponding description for each.
# Comment `--nonIS` if you wish to generate instance-sensitive Cityscapes Edge GT
generate-cityscapes-sbd-gt \
--nonIS \
--only-full-scale \
--nproc 48 \
--root /home/robert.breslin/datasets/cityscapes/ \
#-o ../cityscapes_edges/gtEval/ # This path is relative to the root path on the previous line

generate-cityscapes-hed-gt \
--val_dir /home/robert.breslin/datasets/cityscapes/gtEval/val/ \
--input_suffix _gtProc_raw_edge.png \
--output_suffix _gtProc_raw_hed_edge.png

generate-cityscapes-hed-gt \
--val_dir /home/robert.breslin/datasets/cityscapes/gtEval/val/ \
--input_suffix _gtProc_thin_edge.png \
--output_suffix _gtProc_thin_hed_edge.png
