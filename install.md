# Installation Instructions

This guide explains how to set up the environment and dependencies required to run the software. The setup uses Conda for reproducibility, as it handles both Conda- and pip-installed packages.

## Prerequisites

- **Conda**: Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution) if you don't have it already. Miniconda is recommended for a lighter installation.
- **Git**: Ensure Git is installed to clone the repository.
- **Hardware/Drivers**: 
  - For GPU acceleration (recommended), you need an NVIDIA GPU with compatible drivers. The default setup uses CUDA Toolkit 12.4. Check your CUDA version with `nvcc --version` or `nvidia-smi`.
  - If you don't have a compatible GPU, you can adjust for CPU-only mode (see "Hardware-Specific Adjustments" below).
- **Operating System**: Tested on Linux (x86_64). May work on Windows/macOS with adjustments, but not guaranteed.

## Setup - Must Complete Entire Section

### **Clone the Repository**
```
git clone git@github.com:adossantos21/paper_2.git
```
### **Setup the Conda Environment**

1. **Launch a terminal and enter the current directory:**
   ```
   cd paper_2
   ```

2. **Create the environment:**
   ```
   conda env create -f environment.yml
   ```
   - This will create an environment named `venv_sebnet` with Python 3.8 and all dependencies.
   - If you want a different name, use `conda env create -f environment.yml -n your-env-name`.
   - The process may take several minutes, as it downloads and installs packages from channels like `pytorch`, `nvidia`, `conda-forge`, and `defaults`.
   - If you encounter solver errors (e.g., package conflicts), try running `conda env create -f environment.yml --no-builds` to ignore specific build strings, or update Conda with `conda update conda`.

3. **Activate the Environment:**
   ```
   conda activate venv_sebnet
   ```
4. **Initialize the Repository:**

   Execute the following commands:
    ```
    chmod +x initialize.sh # Makes initialize.sh an executable
    ./initialize.sh # Executes initialize.sh, which installs remaining packages for mmpretrain and mmsegmentation
    ```
    - This will install remaining packages for MMPretrain and MMSegmentation
    - It will also resolve an MMCV Conflict between MMPretrain and MMSegmentation
5. **Verify Installation**:

   Run the following to check PyTorch (should show version 2.4.1 and CUDA status).
   ```
   python -c "import torch; print(torch.__version__)"
   python -c "print('CUDA available:' if torch.cuda.is_available() else 'CUDA not available')"
   ```
   If CUDA is not available but you expect it to be, ensure your NVIDIA drivers are up to date and match your CUDA toolkit version.

   Run the following to check that MMEngine and MMCV were properly installed:
   ```
   python -c "import mmengine; print(mmengine.__version__)"
   python -c "import mmcv; print(mmcv.__version__)"
   ```

6. **Run the Software**:

   Follow the usage instructions in [README.md](https://github.com/adossantos21/paper_2/blob/main/README.md).

## Hardware-Specific Adjustments

The default `environment.yml` is pinned to PyTorch 2.4.1 with CUDA 12.4 for GPU support. If your hardware differs (e.g., older CUDA version, no GPU, or different architecture), you may need to modify the setup to avoid installation failures or runtime errors. Here's how:

### Different CUDA Version (e.g., CUDA 12.1 or 11.8)
- The pinned packages like `pytorch=2.4.1=py3.8_cuda12.4_cudnn9.1.0_0`, `pytorch-cuda=12.4`, `torchaudio=2.4.1=py38_cu124`, and `torchvision=0.19.1=py38_cu124` are specific to CUDA 12.4.
- To adjust:
1. Open `environment.yml` and remove or comment out (with `#`) the following lines under `dependencies`:
    - `pytorch=2.4.1=py3.8_cuda12.4_cudnn9.1.0_0`
    - `pytorch-cuda=12.4=hc786d27_7`
    - `pytorch-mutex=1.0=cuda`
    - `torchaudio=2.4.1=py38_cu124`
    - `torchvision=0.19.1=py38_cu124`
    - Any other CUDA-specific packages like `cuda-cudart`, `cuda-cupti`, etc. (search for "cuda" in the file).
2. Create the environment as usual (step 3 above).
3. Activate the environment.
4. Install PyTorch for your CUDA version using one of these commands (replace with your version; check your CUDA with `nvcc --version`):
   - For CUDA 12.1:
     ```
     conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
     ```
   - For CUDA 11.8:
     ```
     conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
     ```
   - For other versions, visit [PyTorch Get Started](https://pytorch.org/get-started/locally/) and select "Conda" as the package manager to generate the command.
   - Note: PyTorch 2.4.x officially supports CUDA 11.8 and 12.1 out-of-the-box, but builds for 12.4 are available via the same channels. If using a newer/older CUDA, you may need to downgrade PyTorch or install from source.

### CPU-Only (No GPU)

If you don't have an NVIDIA GPU or prefer CPU mode:
1. Follow the steps above to remove/comment out the CUDA-specific packages (as in the CUDA adjustment section).
2. Create and activate the environment.
3. Install the CPU version:
    ```
    conda install pytorch torchvision torchaudio cpuonly -c pytorch
    ```
    - Performance will be slower without GPU acceleration.

### Other Potential Changes
- **Python Version**: The env uses Python 3.8. If you need a different version (e.g., 3.10), change `python=3.8.20=he870216_0` and `python_abi=3.8=2_cp38` in `environment.yml`, then recreate the env. Some packages may need version adjustments.
- **Channels**: If packages fail to download, ensure your Conda is configured to use the listed channels (they are specified in the yml).
- **Pip Packages**: The `pip:` section at the end installs additional libraries. If any fail (rare), install them manually with `pip install <package>` after activation.

If you encounter issues, check the Conda documentation or open an issue in the repository.
