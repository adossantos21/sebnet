FROM ubuntu:20.04

ARG GIT_PAT

ARG KAGGLE_JSON_PATH

ARG SBD_PREDS_DIR

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and Git
RUN apt-get update && apt-get install -y \
    git \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh && \
    /opt/conda/bin/conda clean --all -y && \
    echo "export PATH=/opt/conda/bin:$PATH" >> /etc/profile

ENV PATH /opt/conda/bin:$PATH

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create and activate Conda env
RUN conda create -n venv_pyEdgeEval python=3.8 -y

SHELL ["/bin/bash", "-c"]

RUN echo "source activate venv_pyEdgeEval" >> ~/.bashrc

ENV PATH /opt/conda/envs/venv_pyEdgeEval/bin:$PATH

# Make a new directory called pyEdgeEval
RUN mkdir /pyEdgeEval/sbd

# Set working directory
WORKDIR /pyEdgeEval

# Run initialize.sh equivalent steps
RUN pip install scipy packaging ftfy regex cityscapesscripts && \
    pip install git+https://${GIT_PAT}@github.com/adossantos21/py-edge-eval.git@sebnet

# Download the Cityscapes Dataset
RUN pip install kaggle && \
    mkdir ~/.kaggle && \
    cat ${KAGGLE_JSON_PATH} > ~/.kaggle/kaggle.json && \
    kaggle datasets download -d shuvoalok/cityscapes -p /data && \
    unzip /data/cityscapes.zip -d /data/

# Copy your SBD Directory to the container
COPY ${SBD_PREDS_DIR} /pyEdgeEval/sbd

# Create eval_hed.sh and eval_sbd.sh
RUN cat <<EOF > /pyEdgeEval/eval_hed.sh
#!/bin/bash
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

CMD ["bash"]
