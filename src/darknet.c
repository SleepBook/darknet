#include <time.h>
#include <stdlib.h>
#include <stdio.h>

#include "parser.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include "connected_layer.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#endif

extern void run_yolo(int argc, char **argv);

void visualize(char *cfgfile, char *weightfile)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    visualize_network(net);
#ifdef OPENCV
    cvWaitKey(0);
#endif
}


int main(int argc, char **argv)
{
    if(argc < 2){
        fprintf(stderr, "usage: %s <function>\n", argv[0]);
        return 0;
    }
    gpu_index = find_int_arg(argc, argv, "-i", 0);
    if(find_arg(argc, argv, "-nogpu")) {
        gpu_index = -1;
    }

#ifndef GPU
    gpu_index = -1;
#else
    if(gpu_index >= 0){
        cudaError_t status = cudaSetDevice(gpu_index);
        check_error(status);
    }
#endif
    
    if (0== strcmp(argv[1], "visualize")){
        visualize(argv[2], (argv >3) ? argv[3] : 0);
    } else if (0 != strcmp(argv[1], "yolo")){
    	fprintf(stderr, "Currently This Acceleration Framework Only Support YOLO\n");
    } else {
    	run_yolo(argc, argv);
    }
    return 0;
}

