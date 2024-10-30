#!/bin/bash

set -eux
export DEBIAN_FRONTEND=noninteractive

main() {
    local pkgs=(
        apt-transport-https
        build-essential
        ca-certificates
        ccache
        curl
        gnupg
        htop
        locales
        net-tools
        software-properties-common
        ssh
        sudo
        udev
        unzip
        v4l-utils
        wget
        zlib1g
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
