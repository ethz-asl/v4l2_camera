FROM ros:noetic-ros-core

ARG TZ=Europe/Zurich
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="${PATH}:/opt/hpcx/ompi/bin"

# All apt packages
COPY devcontainer_all_packages.sh /tmp/devcontainer_all_packages.sh
RUN /tmp/devcontainer_all_packages.sh && rm -f /tmp/devcontainer_all_packages.sh

ENV LANG="en_US.UTF-8" \
    LANGUAGE="en_US:en" \
    LC_ALL="en_US.UTF-8"

# Add user "asl" with sudo rights
RUN groupadd -r asl && \
    useradd --create-home --gid asl --groups dialout,plugdev --shell /bin/bash asl && \
    mkdir -p /etc/sudoers.d && \
    echo 'asl ALL=NOPASSWD: ALL' > /etc/sudoers.d/asl

# CUDA: Install
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb && \
    sudo dpkg -i cuda-keyring_1.1-1_all.deb && \
    sudo apt update && \
    sudo apt -y install cuda-toolkit-12-6 cudnn9-cuda-12

# OpenCV 4.10.0 install
COPY build_opencv.sh /tmp/build_opencv.sh
RUN bash /tmp/build_opencv.sh

# Git-prompt: Source
RUN echo 'if [ -f "$HOME/.bash-git-prompt/gitprompt.sh" ]; then GIT_PROMPT_ONLY_IN_REPO=1; source "$HOME/.bash-git-prompt/gitprompt.sh"; fi' >> /home/asl/.bash_aliases

# ROS: Source
RUN echo "source /opt/ros/noetic/setup.bash" >> /home/asl/.bash_aliases

# ROS: rosdep init
RUN sudo rosdep init && \
    rosdep update && \
    rosdep install --from-paths . --ignore-src -r -y

# Clean up
RUN rm -rf /tmp/* /var/lib/apt/lists/* /var/tmp/* /var/cache/apt/archives/*

ENTRYPOINT ["/bin/bash"]
