# v4l2_camera
This repository contains was adapted from [usb_cam](https://github.com/ros-drivers/usb_cam) to run Arducam AR0234 camera's on a Jetson Xavier NX. The AR0234 camera supports mono and Bayer 10-bit output in this [layout](https://www.kernel.org/doc/html/v4.9/media/uapi/v4l/pixfmt-srggb10.html). The driver applies debayering and conversion to RGB8 and outputs the raw image as a ros topic.

## Setup
### Arducam dependencies
In order to run the cameras on the Jetson Xavier NX, the [Arducam quickstart guide](https://docs.arducam.com/Nvidia-Jetson-Camera/Jetvariety-Camera/Quick-Start-Guide/) needs to be followed. Mainly the following steps:

```
1. wget https://github.com/ArduCAM/MIPI_Camera/releases/download/v0.0.3/install_full.sh
2. chmod +x install_full.sh
3. ./install_full.sh -m arducam
4. apt-get install v4l-utils
```

### OpenCV 4.9.X
OpenCV >=4.9.0 is required in order to run the debayering process. An installation scripts inside `.devcontainer` is provided as help on how to compile it from source.
