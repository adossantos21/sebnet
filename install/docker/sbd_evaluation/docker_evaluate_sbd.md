# Return to README.md
[README.md](../../../README.md)

# Installation Instructions

This guide provides the Docker commands to build and correctly launch a container for evaluating SBD metrics. It assumes you have already generated SBD predictions and ground truth labels.

## Prerequisites

- **Install Docker**: Install [Docker](https://docs.docker.com/get-docker/) 
- **Git**:
  - Ensure Git is installed to clone the repository.
  - Ensure you have your personal access token (PAT) on standby. Otherwise, you'll have to configure the Dockerfile for SSH or Github CLI cloning.

## Setup - Must Complete Entire Section

### **Build the Image and Run the Container**

1. **Download the Dockerfile and navigate in your terminal to the directory it was downloaded**
   
   [Dockerfile](Dockerfile)

2. **Copy the SBD Predictions to the directory containing your Dockerfile**

   ```
   cp <rel_path_to_SBD_preds> <abs_path_to_dir_containing_Dockerfile>
   ```

   Alternatively, you can simply move the SBD predictions:

   ```
   mv <rel_path_to_SBD_preds> <abs_path_to_dir_containing_Dockerfile>
   ```
   
3. **Build with PAT**
   ```
   docker build --build-arg GIT_PAT=<your_github_pat> --build-arg KAGGLE_JSON_PATH=<abs_path_to_kaggle_json_file> --build-arg SBD_PREDS_DIR=<abs_path_containing_sbd_predictions> -t pyedgeeval-image .
   ```
   
4. **Run the container:**

   You'll need at least 8 GB of shared memory to evaluate on 8 nproc.
   
   Default run:
   ```
   docker run --shm-size 8g -it pyedgeeval-image
   ```
