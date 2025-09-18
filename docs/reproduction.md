### Return to README.md

[README.md](../README.md)

### Repeating the experiments outlined in the Description section:

#### To pretrain a model:

1. Navigate to the `mmpretrain/` directory:
   ```
   cd mmpretrain/
   ```
   
2. Find the config file path, e.g., `configs/alex_sebnet/pretrain01_tests/pretrain01_staged_1xb64_in1k.py`

3. **(Optional)** If you opted for the [venv](../install/venv/install.md) install approach over [docker](../install/docker/reproduction/docker_reproduction.md), activate the conda environment:
   ```
   conda activate venv_sebnet
   ```
   
4. In `main.sh`, update <config_file_path> with the path from the previous step:
   ```
   export CUDA_VISIBLE_DEVICES=0
   python tools/train.py <config_file_path>
   ```

5. Run the executable:
   ```
   ./main.sh
   ```

#### To finetune a model:

1. Navigate to the `mmsegmentation/` directory:
   ```
   cd mmsegmentation/
   ```
2. Find the config file path, e.g., `configs/sebnet/sebnet_baseline-p-d-sbd-bas-head_1xb6_cityscapes.py`
3. **(Optional)** If you opted for the [venv](../install/venv/install.md) install approach over [docker](../install/docker/reproduction/docker_reproduction.md), activate the conda environment:
   ```
   conda activate venv_sebnet
   ```

##### Single-GPU

4. In `main.sh`, update <config_file_path> with the path from the previous step:
   ```
   export CUDA_VISIBLE_DEVICES=0
   python tools/train.py <config_file_path>
   ```
   
5. Run the executable:
   ```
   ./main.sh
   ```
   
##### Multi-GPU

4. In `main_dist.sh`, update <config_file_path> with the path from the previous step:
   ```
   export CUDA_VISIBLE_DEVICES=0,1,2,3
   export PORT=29500
   export GPUS=4
   ./tools/dist_train.sh \
       <config_file_path> \
       $GPUS
   ```

5. Run the executable:
   ```
   ./main.sh
   ```
