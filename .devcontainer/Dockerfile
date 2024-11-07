# Base Stage: ROS and CUDA Installation
FROM ros:noetic-ros-core AS base

ARG TZ=Europe/Zurich
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="${PATH}:/opt/hpcx/ompi/bin"

# All apt packages
COPY .devcontainer/devcontainer_all_packages.sh /tmp/devcontainer_all_packages.sh
RUN /tmp/devcontainer_all_packages.sh && rm -f /tmp/devcontainer_all_packages.sh

ENV LANG="en_US.UTF-8" \
    LANGUAGE="en_US:en" \
    LC_ALL="en_US.UTF-8"

# Add user "asl" with sudo rights
RUN groupadd -r asl && \
    useradd --create-home --gid asl --groups dialout,plugdev --shell /bin/bash asl && \
    mkdir -p /etc/sudoers.d && \
    echo 'asl ALL=NOPASSWD: ALL' > /etc/sudoers.d/asl

# CUDA: Install (NOTE: libcublas 12.6 is needed for trtexec)
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb && \
    sudo dpkg -i cuda-keyring_1.1-1_all.deb && \
    sudo apt update && \
    sudo apt -y install cuda-toolkit-11-8 cudnn9-cuda-11 libcublas-12-6

# CUDA: Add PATH and LD_LIBRARY_PATH to .bash_aliases
RUN echo 'export PATH=$PATH:/usr/local/cuda-11.8/bin' >> /home/asl/.bash_aliases && \
    echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/lib64' >> /home/asl/.bash_aliases

# TensorRT: Install from .deb file: Seems we run 8.5.2.2 (which is bundled with 8.5.3)
COPY .devcontainer/nv-tensorrt-local-repo-ubuntu2004-8.5.3-cuda-11.8_1.0-1_amd64.deb /tmp/tensorrt.deb
RUN sudo dpkg -i /tmp/tensorrt.deb
RUN sudo cp /var/nv-tensorrt-local-repo-ubuntu2004-8.5.3-cuda-11.8/nv-tensorrt-local-3EFA7C6A-keyring.gpg /usr/share/keyrings/
RUN sudo apt update && sudo apt install -y tensorrt=8.5.2.2-1+cuda11.8 \
                                           libnvinfer8=8.5.2-1+cuda11.8 \
                                           libnvinfer-plugin8=8.5.2-1+cuda11.8 \
                                           libnvparsers8=8.5.2-1+cuda11.8 \
                                           libnvonnxparsers8=8.5.2-1+cuda11.8 \
                                           libnvinfer-bin=8.5.2-1+cuda11.8 \
                                           libnvinfer-dev=8.5.2-1+cuda11.8 \
                                           libnvinfer-plugin-dev=8.5.2-1+cuda11.8 \
                                           libnvparsers-dev=8.5.2-1+cuda11.8 \
                                           libnvonnxparsers-dev=8.5.2-1+cuda11.8 \
                                           libnvinfer-samples=8.5.2-1+cuda11.8

# OpenCV Build Stage
FROM base AS opencv_build

# OpenCV 4.10.0 install
COPY .devcontainer/build_opencv.sh /tmp/build_opencv.sh
RUN bash /tmp/build_opencv.sh && rm -rf /tmp/build_opencv.sh

# Final Stage: Combine ROS, CUDA, and OpenCV
FROM base AS final

# Copy OpenCV installation from the build stage
COPY --from=opencv_build /usr/local /usr/local

# Git-prompt: Source
RUN echo 'if [ -f "$HOME/.bash-git-prompt/gitprompt.sh" ]; then GIT_PROMPT_ONLY_IN_REPO=1; source "$HOME/.bash-git-prompt/gitprompt.sh"; fi' >> /home/asl/.bash_aliases

# ROS: Source
RUN echo "source /opt/ros/noetic/setup.bash" >> /home/asl/.bash_aliases

# ROS: rosdep install (TODO: maybe not so clean?)
RUN sudo rosdep init
USER asl
RUN rosdep update
COPY package.xml /tmp/
RUN rosdep install --from-paths /tmp/ --ignore-src -r -y -v
USER root

# Clean up
RUN rm -rf /tmp/* /var/lib/apt/lists/* /var/tmp/* /var/cache/apt/archives/*
ENTRYPOINT ["/bin/bash"]
