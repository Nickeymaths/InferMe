# Dev image: full CUDA toolkit (nvcc, cuda-gdb, nsight-compute CLI libs)
# Driver supports CUDA 13.2; 12.x images also run fine. Change tag if needed.
ARG CUDA_TAG=12.9.1-cudnn-devel-ubuntu24.04
FROM nvidia/cuda:${CUDA_TAG}

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build gdb git curl vim ca-certificates \
      python3 python3-pip python3-venv clang-format \
    && rm -rf /var/lib/apt/lists/*

# Match host UID/GID so files in the bind mount aren't root-owned
ARG UID=1000
ARG GID=1000
RUN (id -nu ${UID} 2>/dev/null | xargs -r userdel -r || true) && \
    (getent group ${GID} | cut -d: -f1 | xargs -r groupdel || true) && \
    groupadd -g ${GID} dev && \
    useradd -m -u ${UID} -g dev -s /bin/bash dev
USER dev
WORKDIR /workspace
