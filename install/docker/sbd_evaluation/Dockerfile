# Use the specified base image
FROM ubuntu:20.04

# Define build arguments
ARG GIT_PAT
ARG SBD_PREDS_DIR

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/opt/conda/bin:$PATH

# Install system dependencies and clean up
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        wget \
        unzip \
        libgl1-mesa-glx \
        libglib2.0-0 \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh && \
    conda clean --all -y && \
    echo "export PATH=/opt/conda/bin:$PATH" >> /etc/profile

# Configure Conda channels (assuming 'tos accept' is intended; if typo, adjust accordingly)
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create and activate Conda environment
RUN conda create -n venv_pyEdgeEval python=3.8 -y

# Switch to bash shell for subsequent RUN commands
SHELL ["/bin/bash", "-c"]

# Update PATH for the Conda environment
ENV PATH=/opt/conda/envs/venv_pyEdgeEval/bin:$PATH

# Set working directory and create necessary directories
WORKDIR /pyEdgeEval
RUN mkdir -p /pyEdgeEval/data/cityscapes /pyEdgeEval/sbd_preds

# Install Python dependencies via pip
RUN pip install --no-cache-dir \
    cityscapesscripts \
    matplotlib \
    mat73 \
    numpy \
    scipy \
    scikit-image \
    prettytable \
    pymatreader \
    opencv-python

# Install PyEdgeEval from GitHub using secret
RUN --mount=type=secret,id=git_pat,dst=/tmp/gitpat.txt \
    GIT_PAT=$(cat /tmp/gitpat.txt) && \
    pip install git+https://${GIT_PAT}@github.com/adossantos21/py-edge-eval.git@sebnet

# Download and extract Cityscapes Dataset using secrets
RUN --mount=type=secret,id=cityscapes_username,dst=/tmp/username.txt \
    --mount=type=secret,id=cityscapes_password,dst=/tmp/password.txt \
    USERNAME=$(cat /tmp/username.txt) && \
    PASSWORD=$(cat /tmp/password.txt) && \
    wget --keep-session-cookies --save-cookies=/tmp/cookies.txt --post-data "username=$USERNAME&password=$PASSWORD&submit=Login" https://www.cityscapes-dataset.com/login/ && \
    wget --load-cookies /tmp/cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1 -P /pyEdgeEval/data/cityscapes/ && \
    rm /tmp/cookies.txt && \
    unzip /pyEdgeEval/data/cityscapes/gtFine_trainvaltest.zip -d /pyEdgeEval/data/cityscapes/ && \
    rm /pyEdgeEval/data/cityscapes/gtFine_trainvaltest.zip

# Generate SBD and HED Ground Truth
RUN cat <<EOF > /pyEdgeEval/generate_sbd_hed_gt.sh
#!/bin/bash
generate-cityscapes-sbd-gt \\
--nonIS \\
--only-full-scale \\
--nproc 48 \\
--root /pyEdgeEval/data/cityscapes/

generate-cityscapes-hed-gt \\
--val_dir /pyEdgeEval/data/cityscapes/gtEval/val/ \\
--input_suffix _gtProc_raw_edge.png \\
--output_suffix _gtProc_raw_hed_edge.png

generate-cityscapes-hed-gt \\
--val_dir /pyEdgeEval/data/cityscapes/gtEval/val/ \\
--input_suffix _gtProc_thin_edge.png \\
--output_suffix _gtProc_thin_hed_edge.png
EOF

RUN chmod +x /pyEdgeEval/generate_sbd_hed_gt.sh && \
    /pyEdgeEval/generate_sbd_hed_gt.sh

# Copy SBD predictions directory
RUN --mount=type=bind,source=${SBD_PREDS_DIR},target=/tmp/sbd_dst \
    cp -r /tmp/sbd_dst/* /pyEdgeEval/sbd_preds/

# Create evaluation scripts
RUN cat <<EOF > /pyEdgeEval/eval_hed.sh
#!/bin/bash
cityscapes-raw-eval \\
<path_to_cityscapes_root> \\
<path_to_hed_predictions> \\
--output-path <output_path> \\
--nonIS \\
--pre-seal \\
--remove-root \\
--categories '1' \\
--max-dist 0.02 \\
--thresholds 99 \\
--nproc 8 \\
--split 'val' \\
--pred-suffix '_SBD.png'
EOF

RUN cat <<EOF > /pyEdgeEval/eval_sbd.sh
#!/bin/bash
cityscapes-raw-eval \\
<path_to_cityscapes_root> \\
<path_to_sbd_predictions> \\
--output-path <output_path> \\
--nonIS \\
--pre-seal \\
--remove-root \\
--multi-label \\
--categories '[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]' \\
--max-dist 0.02 \\
--thresholds 99 \\
--nproc 8 \\
--split 'val' \\
--pred-suffix '_SBD.png'
EOF

# Make evaluation scripts executable
RUN chmod +x /pyEdgeEval/eval_hed.sh /pyEdgeEval/eval_sbd.sh

# Create non-root user and adjust permissions
RUN useradd -m -s /bin/bash appuser && \
    chown -R appuser:appuser /opt/conda /pyEdgeEval && \
    echo "source activate venv_pyEdgeEval" >> /home/appuser/.bashrc

# Switch to non-root user
USER appuser

# Default command
CMD ["bash"]
