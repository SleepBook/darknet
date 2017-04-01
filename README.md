## Intro

This project aim to realize and optimize the yolo-tiny network on embedded platforms. These platforms including FPGA(Zynq), embedded GPU(Nvidia Jetson)

The project only has the inference functionality of YOLO-Tiny, the training functionality is not reserved, since it's not likly you want to train your network on these platforms.

The project derived from the Darknet framework, credit goes there. 

## Usage
To test on an image:
    ./darknet yolo test cfg/yolo-tiny.cfg yolo-tiny.weights IMAGEPATH

To test on video:
    .darknet yolo demo cfg/yolo-tiny.cfg yolo-tiny.weights VIDEOPATH

To test on webcame:
    .darknet yolo demo cfg/yolo-tiny.cfg yolo-tiny.weight

To visualize the net:
    ./darknet visualize cfgfile weightfile
