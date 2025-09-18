FROM ubuntu:20.04

ARG GIT_PAT

ARG SBD_PREDS_DIR

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and Git
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh && \
    /opt/conda/bin/conda clean --all -y && \
    echo "export PATH=/opt/conda/bin:$PATH" >> /etc/profile

ENV PATH=/opt/conda/bin:$PATH

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create and activate Conda env
RUN conda create -n venv_pyEdgeEval python=3.8 -y

SHELL ["/bin/bash", "-c"]

ENV PATH=/opt/conda/envs/venv_pyEdgeEval/bin:$PATH

# Make a new directory called pyEdgeEval
RUN mkdir /pyEdgeEval

RUN mkdir /pyEdgeEval/data

RUN mkdir /pyEdgeEval/data/cityscapes

RUN mkdir /pyEdgeEval/sbd_preds

# Set working directory
WORKDIR /pyEdgeEval

# Run initialize.sh equivalent steps
RUN pip install cityscapesscripts matplotlib mat73 numpy scipy scikit-image prettytable pymatreader opencv-python

# Install PyEdgeEval
RUN --mount=type=secret,id=git_pat,dst=/tmp/gitpat.txt \
    GIT_PAT=$(cat /tmp/gitpat.txt) && \
    pip install git+https://${GIT_PAT}@github.com/adossantos21/py-edge-eval.git@sebnet

# Download and install the Cityscapes Dataset
RUN --mount=type=secret,id=cityscapes_username,dst=/tmp/username.txt \
    --mount=type=secret,id=cityscapes_password,dst=/tmp/password.txt \
    USERNAME=$(cat /tmp/username.txt) && \
    PASSWORD=$(cat /tmp/password.txt) && \
    wget --keep-session-cookies --save-cookies=/tmp/cookies.txt --post-data "username=$USERNAME&password=$PASSWORD&submit=Login" https://www.cityscapes-dataset.com/login/ && \
    wget --load-cookies /tmp/cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1 -P /pyEdgeEval/data/cityscapes/ && \
    rm /tmp/cookies.txt
    
RUN unzip /pyEdgeEval/data/cityscapes/gtFine_trainvaltest.zip -d /pyEdgeEval/data/cityscapes/ && \
    rm /pyEdgeEval/data/cityscapes/gtFine_trainvaltest.zip

# Generate SBD Ground Truth
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

RUN chmod +x /pyEdgeEval/generate_sbd_hed_gt.sh

RUN /pyEdgeEval/generate_sbd_hed_gt.sh

# Copy your SBD Directory to the container
RUN --mount=type=bind,source=${SBD_PREDS_DIR},target=/tmp/sbd_dst \
    cp -r /tmp/sbd_dst/* /pyEdgeEval/sbd_preds/

# Create eval_hed.sh and eval_sbd.sh
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

# Make the scripts executable
RUN chmod +x /pyEdgeEval/eval_hed.sh

RUN chmod +x /pyEdgeEval/eval_sbd.sh

# Create a non-root user and group
RUN useradd -m -s /bin/bash appuser

# Adjust permissions on key directories/files so the non-root user can access/read/write/execute them
# (Conda env, working dir, datasets, etc.; use -R for recursive)
RUN chown -R appuser:appuser /opt/conda /pyEdgeEval

# Set up Conda activation for the non-root user's interactive shell
RUN echo "source activate venv_pyEdgeEval" >> /home/appuser/.bashrc

# Switch to the non-root user for runtime
USER appuser

CMD ["bash"]
