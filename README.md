This Project Drived from the Darknet Project, which you can find the information below. 

Our aim is to accelerate the Yolo-tiny NN on Embedded Platforms, so only the code relevant to Yolo-tiny inference is kept. 


#Darknet#
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).

To test on an image:
    ./darknet yolo test cfg/yolo-tiny.cfg yolo-tiny.weights IMAGEPATH

To test on video:
    .darknet yolo demo cfg/yolo-tiny.cfg yolo-tiny.weights VIDEOPATH

To test on webcame:
    .darknet yolo demo cfg/yolo-tiny.cfg yolo-tiny.weight

To visualize the net:
    ./darknet visualize cfgfile weightfile
