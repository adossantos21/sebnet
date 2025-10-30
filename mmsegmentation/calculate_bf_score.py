from tools.analysis_tools.calculate_bfscore import BFScore

gt_path = '/home/robert.breslin/cityscapes_gt_dir'
preds_path = '/home/robert.breslin/alessandro/testing/paper_2/mmsegmentation/preds/dense/pidnet-l/raw'

metrics = BFScore(
    pred_path=preds_path, 
    mask_path=gt_path,
    num_classes=19,
    num_proc=10,
    image_height=1024,
    image_width=2048,
    px_tolerance=3
    )
results = metrics.forward()
print(results)