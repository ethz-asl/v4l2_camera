#!/usr/bin/env bash
# 2019 Michael de Gans

set -e

# change default constants here:
readonly PREFIX=/usr/local  # install prefix, (can be ~/.local for a user install)
readonly DEFAULT_VERSION=4.10.0  # controls the default version (gets reset by the first argument)
readonly CPUS=$(nproc)  # controls the number of jobs
readonly BUILD_DIR="/tmp"

# better board detection. if it has 6 or more cpus, it probably has a ton of ram too
if [[ $CPUS -gt 5 ]]; then
    # something with a ton of ram
    JOBS=$CPUS
else
    JOBS=1  # you can set this to 4 if you have a swap file
    # otherwise a Nano will choke towards the end of the build
fi

cleanup () {
# https://stackoverflow.com/questions/226703/how-do-i-prompt-for-yes-no-cancel-input-in-a-linux-shell-script
    while true ; do
        echo "Do you wish to remove temporary build files in $BUILD_DIR/build_opencv ? "
        if ! [[ "$1" -eq "--test-warning" ]] ; then
            echo "(Doing so may make running tests on the build later impossible)"
        fi
        # read -p "Y/N " yn
    yn="n"
        case ${yn} in
            [Yy]* ) rm -rf $BUILD_DIR/build_opencv ; break;;
            [Nn]* ) break ;;
            * ) echo "Please answer yes or no." ;;
        esac
    done
}

setup () {
    cd $BUILD_DIR
    if [[ -d "$BUILD_DIR/build_opencv" ]] ; then
        echo "It appears an existing build exists in $BUILD_DIR/build_opencv"
        cleanup
    fi
    mkdir -p build_opencv
    cd build_opencv
}

git_source () {
    echo "Getting version '$1' of OpenCV"
    git clone --depth 1 --branch "$1" https://github.com/opencv/opencv.git
    git clone --depth 1 --branch "$1" https://github.com/opencv/opencv_contrib.git
}

install_dependencies () {
    # open-cv has a lot of dependencies, but most can be found in the default
    # package repository or should already be installed (eg. CUDA).
    echo "Installing build dependencies."
    sudo apt-get update
    sudo apt-get dist-upgrade -y --autoremove
    sudo apt-get install -y \
        build-essential \
        cmake \
        git \
        gfortran \
        libatlas-base-dev \
        libavcodec-dev \
        libavformat-dev \
        libavresample-dev \
        libcanberra-gtk3-module \
        libdc1394-22-dev \
        libeigen3-dev \
        libglew-dev \
        libgstreamer-plugins-base1.0-dev \
        libgstreamer-plugins-good1.0-dev \
        libgstreamer1.0-dev \
        libgtk-3-dev \
        libjpeg-dev \
        libjpeg8-dev \
        libjpeg-turbo8-dev \
        liblapack-dev \
        liblapacke-dev \
        libopenblas-dev \
        libpng-dev \
        libpostproc-dev \
        libswscale-dev \
        libtbb-dev \
        libtbb2 \
        libtesseract-dev \
        libtiff-dev \
        libv4l-dev \
        libxine2-dev \
        libxvidcore-dev \
        libx264-dev \
        pkg-config \
        python-dev \
        python-numpy \
        python3-dev \
        python3-numpy \
        python3-matplotlib \
        qv4l2 \
        v4l-utils \
        zlib1g-dev
}

# Automatically detect installed CUDA version
detect_cuda_version() {
    if command -v /usr/local/cuda/bin/nvcc &> /dev/null; then
        CUDA_VERSION=$(/usr/local/cuda/bin/nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
    else
        echo "CUDA not found. Please install CUDA before running this script."
        exit 1
    fi
}

# Automatically detect installed cuDNN version
detect_cudnn_version() {
    if [[ -f "/usr/include/cudnn_version.h" ]]; then
        CUDNN_MAJOR=$(grep -oP '(?<=#define CUDNN_MAJOR )\d+' /usr/include/cudnn_version.h)
        CUDNN_MINOR=$(grep -oP '(?<=#define CUDNN_MINOR )\d+' /usr/include/cudnn_version.h)
        CUDNN_PATCHLEVEL=$(grep -oP '(?<=#define CUDNN_PATCHLEVEL )\d+' /usr/include/cudnn_version.h)
        CUDNN_VERSION="${CUDNN_MAJOR}.${CUDNN_MINOR}.${CUDNN_PATCHLEVEL}"
    else
        echo "cuDNN not found. Please install cuDNN before running this script."
        exit 1
    fi
}

# Set CUDA compute capabilities based on detected CUDA version
set_cuda_arch_bin() {
    case "$CUDA_VERSION" in
        11.*)
            CUDA_ARCH_BIN="5.3,6.1,6.2,7.0,7.5,8.0"
            ;;
        12.*)
            CUDA_ARCH_BIN="6.1,6.2,7.5,8.0,8.6,8.7"
            ;;
        *)
            echo "Unsupported CUDA version detected: $CUDA_VERSION"
            exit 1
            ;;
    esac
}

# Update the configure function to use dynamic CUDA and cuDNN version detection
configure() {
    detect_cuda_version
    detect_cudnn_version
    set_cuda_arch_bin

    local CMAKEFLAGS="
        -D BUILD_EXAMPLES=OFF
        -D BUILD_opencv_python2=ON
        -D BUILD_opencv_python3=ON
        -D CMAKE_BUILD_TYPE=RELEASE
        -D CMAKE_INSTALL_PREFIX=${PREFIX}
        -D CUDA_ARCH_BIN=${CUDA_ARCH_BIN}
        -D CUDA_ARCH_PTX=
        -D CUDA_FAST_MATH=ON
        -D CUDNN_VERSION='${CUDNN_VERSION}'
        -D EIGEN_INCLUDE_PATH=/usr/include/eigen3 
        -D ENABLE_NEON=OFF
        -D OPENCV_DNN_CUDA=ON
        -D OPENCV_ENABLE_NONFREE=ON
        -D OPENCV_EXTRA_MODULES_PATH=$BUILD_DIR/build_opencv/opencv_contrib/modules
        -D OPENCV_GENERATE_PKGCONFIG=ON
        -D WITH_CUBLAS=ON
        -D WITH_CUDA=ON
        -D WITH_CUDNN=ON
        -D WITH_GSTREAMER=ON
        -D WITH_LIBV4L=ON
        -D WITH_OPENGL=ON"

    if [[ "$1" != "test" ]] ; then
        CMAKEFLAGS="
        ${CMAKEFLAGS}
        -D BUILD_PERF_TESTS=OFF
        -D BUILD_TESTS=OFF"
    fi

    echo "CUDA version: $CUDA_VERSION"
    echo "cuDNN version: $CUDNN_VERSION"
    echo "CUDA_ARCH_BIN: $CUDA_ARCH_BIN"
    echo "cmake flags: ${CMAKEFLAGS}"
    cd $BUILD_DIR/build_opencv/opencv
    mkdir -p build
    cd build
    cmake ${CMAKEFLAGS} .. 2>&1 | tee -a configure.log
}

main () {

    local VER=${DEFAULT_VERSION}

    # parse arguments
    if [[ "$#" -gt 0 ]] ; then
        VER="$1"  # override the version
    fi

    if [[ "$#" -gt 1 ]] && [[ "$2" == "test" ]] ; then
        DO_TEST=1
    fi

    # prepare for the build:
    setup
    install_dependencies
    # if [[ ! -d "$BUILD_DIR/build_opencv" ]] ; then
    git_source ${VER}
    # fi

    if [[ ${DO_TEST} ]] ; then
        configure test
    else
        configure
    fi

    # start the build
    make -j${JOBS} 2>&1 | tee -a build.log

    if [[ ${DO_TEST} ]] ; then
        make test 2>&1 | tee -a test.log
    fi

    # avoid a sudo make install (and root owned files in ~) if $PREFIX is writable
    if [[ -w ${PREFIX} ]] ; then
        make install 2>&1 | tee -a install.log
    else
        sudo make install 2>&1 | tee -a install.log
    fi

    cleanup --test-warning

}

main "$@"
