# Return to README.md
[README.md](../../../README.md)

# Installation Instructions

The benefits of this guide is that the Dockerfile builds all the required software and resolves the bugs described in [install.md](../../virt_env/install.md) in a single step.

This guide provides the Docker commands to build and correctly launch a container for reproduction activities such as:
- Training your own models
- Creating dense or SBD predictions
- Creating your SBD ground truths
- Evaluating dense (regular, e.g., mIoU) and sparse (SBD/HED, e.g., mF at ODS) metrics

## Prerequisites

- **Install Docker and NVIDIA Container Toolkit**: Install [Docker](https://docs.docker.com/get-docker/) and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (for GPU support).
- **Git**:
  - Ensure Git is installed to clone the repository.
  - Ensure you have your personal access token (PAT) on standby. Otherwise, you'll have to configure the Dockerfile for SSH or Github CLI cloning.
- **Hardware/Drivers**: 
  - For GPU acceleration (required for Docker install), you need an NVIDIA GPU with compatible drivers. Driver installation is part of the `Versioned Online Documentation` guide, referenced at the [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive).

## Setup - Must Complete Entire Section

### **Build the Image and Run the Container**

1. **Download the Dockerfile and navigate in your terminal to the directory it was downloaded**
   
   [Dockerfile](Dockerfile)

2. **Build with PAT**
   ```
   docker build --build-arg GIT_PAT=<your_github_pat> -t sebnet-env .
   ```
   
3. **Run the container:**

   You'll need at least 8 GB of shared memory to run a training.
   
   Default run:
   ```
   docker run --shm-size 8g --gpus all -it sebnet-env
   ```

   To run with a mounted volume:
   ```
   docker run --shm-size 8g --gpus all -v <abs_path_to_volume>:<abs_path_on_container> -it sebnet-env
   ```

   For example, if you were to mount a dataset directory from your machine to the container:
   ```
   docker run --shm-size 8g --gpus all -v /home/username/datasets/:/sebnet/mmsegmentation/data/ -it sebnet-env
   ```
