FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04

ARG GIT_PAT

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and Git
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
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
RUN conda create -n venv_sebnet python=3.8 -y

SHELL ["/bin/bash", "-c"]

RUN echo "source activate venv_sebnet" >> ~/.bashrc

ENV PATH /opt/conda/envs/venv_sebnet/bin:$PATH

# Clone repo using HTTPS with PAT
RUN git clone https://${GIT_PAT}@github.com/adossantos21/sebnet.git /sebnet

WORKDIR /sebnet

# Install PyTorch (CUDA 12.1 example; adjust for 12.4)
RUN pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# Run initialize.sh equivalent steps
RUN pip install scipy packaging ftfy regex cityscapesscripts && \
    pip install -U openmim && \
    cd mmpretrain && mim install -e . && cd .. && \
    cd mmsegmentation && pip install -v -e . && cd .. && \
    pip install git+https://${GIT_PAT}@github.com/adossantos21/py-edge-eval.git@sebnet

# Apply bug fixes (manual edits)
RUN sed -i 's/2, 3/2, 3, 4/' /opt/conda/envs/venv_sebnet/lib/python3.8/site-packages/mmengine/structures/pixel_data.py && \
    sed -i 's/The dim of value must be 2 or 3, but got {value.ndim}/The dim of value must be 2, 3, or 4, but got {value.ndim}/' /opt/conda/envs/venv_sebnet/lib/python3.8/site-packages/mmengine/structures/pixel_data.py && \
    sed -i 's/losses = self._run_forward(data, mode='"'"'loss'"'"')/losses, logits = self._run_forward(data, mode='"'"'loss'"'"')/' /opt/conda/envs/venv_sebnet/lib/python3.8/site-packages/mmengine/model/wrappers/distributed.py && \
    sed -i 's/return log_vars/return log_vars, logits/' /opt/conda/envs/venv_sebnet/lib/python3.8/site-packages/mmengine/model/wrappers/distributed.py

# Verify (optional, for build testing)
RUN python -c "import torch; print(torch.__version__, torch.cuda.is_available())" && \
    python -c "import mmengine; print(mmengine.__version__)" && \
    python -c "import mmcv; print(mmcv.__version__)" && \
    python -c "import mmpretrain; print(mmpretrain.__version__)" && \
    python -c "import mmseg; print(mmseg.__version__)"

# Create a non-root user and group
RUN useradd -m -s /bin/bash appuser

# Adjust permissions on key directories/files so the non-root user can access/read/write/execute them
# (Conda env, working dir, datasets, etc.; use -R for recursive)
RUN chown -R appuser:appuser /opt/conda /sebnet

# Set up Conda activation for the non-root user's interactive shell
RUN echo "source activate venv_pyEdgeEval" >> /home/appuser/.bashrc

# Switch to the non-root user for runtime
USER appuser

CMD ["bash"]
