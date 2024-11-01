#!/bin/bash

set -eux
export DEBIAN_FRONTEND=noninteractive

main() {
    local pkgs=(
        apt-transport-https
        build-essential
        ca-certificates
        ccache
        cmake
        curl
        gfortran
        git
        gnupg
        htop
        libatlas-base-dev
        libavcodec-dev
        libavformat-dev
        libavresample-dev
        libcanberra-gtk3-module
        libdc1394-22-dev
        libeigen3-dev
        libglew-dev
        libgstreamer-plugins-base1.0-dev
        libgstreamer-plugins-good1.0-dev
        libgstreamer1.0-dev
        libgtk-3-dev
        libjpeg-dev
        libjpeg-turbo8-dev
        libjpeg8-dev
        liblapack-dev
        liblapacke-dev
        libopenblas-dev
        libpng-dev
        libpostproc-dev
        libswscale-dev
        libtbb-dev
        libtbb2
        libtesseract-dev
        libtiff-dev
        libv4l-dev
        libx264-dev
        libxine2-dev
        libxvidcore-dev
        locales
        net-tools
        pkg-config
        python-dev
        python-numpy
        python3-dev
        python3-matplotlib
        python3-numpy
        python3-rosdep
        qv4l2
        software-properties-common
        ssh
        sudo
        udev
        unzip
        v4l-utils
        vim
        wget
        zlib1g
        zlib1g-dev
    )

    apt-get update
    apt-get upgrade -y
    apt-get -y --quiet --no-install-recommends install "${pkgs[@]}"

    mkdir -p /root/.ssh \
        && chmod 0700 /root/.ssh \
        && ssh-keyscan github.com > /root/.ssh/known_hosts

    ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime && echo ${TZ} > /etc/timezone
    sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen
    locale-gen en_US.UTF-8
    dpkg-reconfigure locales

    apt-get -y autoremove
    apt-get clean autoclean
    rm -rf /var/lib/apt/lists/{apt,dpkg,cache,log} /tmp/* /var/tmp/*
}

main "$@"
