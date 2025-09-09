# Use `-h` flag to view all optional arguments and corresponding description for each.
# Comment `--nonIS` if you wish to generate instance-sensitive Cityscapes Edge GT
generate-cityscapes-gt \
--nonIS \
--only-full-scale \
--nproc 8 \
--root /home/robert.breslin/datasets/cityscapes/ \
-o ../cityscapes_edges/ # This path is relative to the root path on the previous line

python generate_hed_gt.py \
--val_dir /home/robert.breslin/datasets/cityscapes_edges/gtEval/val/ \
--input_suffix _gtProc_raw_edge.png \
--output_suffix _gtProc_raw_hed_edge.png

python generate_hed_gt.py \
--val_dir /home/robert.breslin/datasets/cityscapes_edges/gtEval/val/ \
--input_suffix _gtProc_thin_edge.png \
--output_suffix _gtProc_thin_hed_edge.png