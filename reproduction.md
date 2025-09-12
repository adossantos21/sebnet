### Repeating the experiments outlined in the Description section:
**To pretrain a model:**
1. Navigate to the `mmpretrain/` directory:
   ```
   cd mmpretrain/
   ```
2. Find the config file path, e.g., `configs/alex_sebnet/pretrain01_tests/pretrain01_staged_1xb64_in1k.py`
3. Update <config_file_path> with the path from the previous step in `./main.sh`, which should look something like:
   ```
   export CUDA_VISIBLE_DEVICES=0
   python tools/train.py <config_file_path>
   ```
4. Activate the conda environment you created from [install.md](https://github.com/adossantos21/paper_2/blob/main/install.md):
   ```
   conda activate venv_sebnet
   ```
5. Run the executable:
   ```
   ./main.sh
   ```

**To finetune a model:**
1. Navigate to the `mmsegmentation/` directory:
   ```
   cd mmsegmentation/
   ```
2. Find the config file path, e.g., `configs/sebnet/sebnet_baseline-p-d-sbd-bas-head_1xb6_cityscapes.py`
3. Update <config_file_path> with the path from the previous step in `./main.sh`, which should look something like:
   ```
   export CUDA_VISIBLE_DEVICES=0
   python tools/train.py <config_file_path>
   ```
4. Activate the conda environment you created from [install.md](https://github.com/adossantos21/paper_2/blob/main/install.md):
   ```
   conda activate venv_sebnet
   ```
5. Run the executable:
   ```
   ./main.sh
   ```
