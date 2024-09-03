ARG PYTORCH="2.2.2"
ARG CUDA="12.1"
ARG CUDA_ALIAS="121"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel
# FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel

RUN DEBIAN_FRONTEND=noninteractive apt-get -y update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y git sudo ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* 


ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID


RUN groupadd --gid $USER_GID $USERNAME
RUN useradd --uid $USER_UID --gid $USER_GID -m $USERNAME
RUN echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME
RUN chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME

ENV SHELL /bin/bash
ENV PATH="${PATH}:/home/${USERNAME}/.local/bin"
