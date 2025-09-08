# Use `-h` flag to view all optional arguments and corresponding description for each.
# Comment `--nonIS` if you wish to generate instance-sensitive Cityscapes Edge GT
generate-cityscapes-gt \
--nonIS \
--only-full-scale \
--nproc 8 \
--root /home/robert.breslin/datasets/cityscapes/ \
-o ../cityscapes_edges/ # This path is relative to the root path on the previous line
