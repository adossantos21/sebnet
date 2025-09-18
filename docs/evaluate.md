### Return to README.md
[README.md](../README.md)

### To download the weights and evaluate performance:
1. If not already downloaded, download the weights:

   **Without Mapillary Pre-training:**
   | Model (Cityscapes)              | Val mIoU (%)                                                                  | Test mIoU (%)                                                                    |  FPS  |
   |---------------------------------|-------------------------------------------------------------------------------|----------------------------------------------------------------------------------|-------|
   | BaselinePDSBDBASHead            | [81.1](https://github.com/adossantos21/paper_2/raw/main/mmsegmentation/work_dirs/sebnet_baseline-p-d-sbd-bas-head_1xb6_cityscapes/20250906_102604/checkpoints/sebnet_baseline-p-d-sbd-bas-head_1xb6_cityscapes/20250906_102604/best_mIoU.pth) | [Pending](https://github.com/<your-username>/<your-repo>/raw/main/largefile.ext) |  31.1 |
   | ConditionalBaselinePSBDBASHead  | [80.7](https://github.com/adossantos21/paper_2/raw/main/mmsegmentation/work_dirs/sebnet_baseline-p-sbd-bas-head-conditioned_1xb6_cityscapes/20250906_102650/checkpoints/sebnet_baseline-p-sbd-bas-head-conditioned_1xb6_cityscapes/20250906_102650/best_mIoU.pth) | [Pending](https://github.com/<your-username>/<your-repo>/raw/main/otherfile.ext) |  35.4 |
   | BaselinePDBASHead (PIDNet, Ours)      | [80.5](https://github.com/adossantos21/paper_2/raw/main/mmsegmentation/work_dirs/sebnet_baseline-p-d-bas-head_1xb6_cityscapes/20250906_105242/checkpoints/sebnet_baseline-p-d-bas-head_1xb6_cityscapes/20250906_105242/best_mIoU.pth)                              | [Pending](https://github.com/adossantos21/paper_2)                         |  31.1 |
   
3. **(Optional)** If you opted for the [venv](../install/venv/install.md) install approach over [docker](../install/docker/reproduction/docker_reproduction.md), you'll have to activate the conda environment:

   ```
   conda activate venv_sebnet
   ```
   
#### For semantic edge detection (SBD) evaluation
1. Enter the `mmsegmentation` directory:
   ```
   cd mmsegmentation/
   ```
2. Generate your SBD predictions
    - Configure `generate_cityscapes_preds.sh`
        - Set the model config file
        - Set the relevant model's checkpoint
        - Set the output path for your SBD predictions
    - Run the executable:
      ```
      ./generate_cityscapes_preds.sh
      ```
3. Generate your SBD ground truth
    - Your environment will have installed the packages required for the following shell commands.
    - Configure `generate_cityscapes_gt.sh`
        - Set the root path of your cityscapes dataset
        - (Optional) - Set the output path *relative* to the cityscapes dataset root path
    - Run the executable:
      ```
      ./generate_cityscapes_gt.sh
      ```

4. Configure `eval_hed.sh` and/or `eval_sbd.sh`
    - Add `-h` flag to see optional arguments and corresponding descriptions.
    - HED Example:
      ```
      cityscapes-raw-eval \
      <path_to_cityscapes_root> \
      <path_to_hed_predictions> \
      --output-path <output_path> \
      --nonIS \
      --pre-seal \
      --remove-root \
      --categories '1' \
      --max-dist 0.02 \
      --thresholds 99 \
      --nproc 8 \
      --split 'val' \
      --pred-suffix '_SBD.png'
      ```
    - SBD Example:
      ```
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
      --pred-suffix '_SBD.png'
      ```
5. Run the executable:
   ```
   ./eval_hed.sh
   ```
   ```
   ./eval_sbd.sh
   ```
#### For image classification or semantic segmentation evaluation
1. Navigate to either `mmpretrain/` or `mmsegmentation/` directory.
2. Find the corresponding config file path, e.g., `configs/sebnet/sebnet_baseline-p-d-sbd-bas-head_1xb6_cityscapes.py`
3. In `./eval.sh`, update <config_file_path> with the path from the previous step. Also, add the checkpoint weights `.pth` file to `eval.sh`, which should look something like:
   ```
   export CUDA_VISIBLE_DEVICES=0
   python tools/test.py \
       <config_file_path> \
       <ckpt.pth>
   ```
4. Run the executable:
   ```
   ./eval.sh
   ```
