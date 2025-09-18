# Return to README.md
[README.md](../../../README.md)

# Installation Instructions

This guide provides the Docker commands to build and correctly launch a container for evaluating SBD metrics. It assumes you have already generated SBD predictions and ground truth labels.

## Prerequisites

- **Install Docker**: Install [Docker](https://docs.docker.com/get-docker/) 
- **Git**:
  - Ensure Git is installed to clone the repository.
  - Ensure you have your personal access token (PAT) on standby. Otherwise, you'll have to configure the Dockerfile for SSH or Github CLI cloning.
  - Have an account with https://cityscapes-dataset.com, and access to your username and password

## Setup - Must Complete Entire Section

### **Build the Image and Run the Container**

1. **Download the Dockerfile and navigate in your terminal to the directory it was downloaded**
   
   [Dockerfile](Dockerfile)

2. **Copy the SBD Predictions to the directory containing your Dockerfile (your predictions must be in this directory for the image to build)**

   ```
   cp <abs_path_to_SBD_preds> <abs_path_to_dir_containing_Dockerfile>
   ```

   Alternatively, you can simply move the SBD predictions:

   ```
   mv <abs_path_to_SBD_preds> <abs_path_to_dir_containing_Dockerfile>
   ```
   
3. **Export your Github PAT, and your Cityscapes username and password to environment variables**

   ```
   export MY_GIT_PAT=<your_github_pat>
   export MY_CITYSCAPES_USERNAME=<your_cityscapes_username>
   export MY_CITYSCAPES_PASSWORD=<your_cityscapes_password>
   ```
   
5. **Build your image**

   ```
   docker build --build-arg SBD_PREDS_DIR=<abs_path_to_SBD_preds> --secret id=git_pat,env=MY_GIT_PAT --secret id=cityscapes_username,env=MY_CITYSCAPES_USERNAME --secret id=cityscapes_password,env=MY_CITYSCAPES_PASSWORD -t pyedgeeval-image .
   ```
   
7. **Run the container:**

   You'll need at least 10 GB of shared memory to evaluate on 8 nproc.
   
   Default run:
   ```
   docker run --shm-size 10g -it pyedgeeval-image
   ```
