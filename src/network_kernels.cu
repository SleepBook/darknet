#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include <stdio.h>
#include <time.h>
#include <assert.h>

#include "network.h"
#include "image.h"
#include "data.h"
#include "utils.h"
#include "parser.h"

#include "crop_layer.h"
#include "connected_layer.h"
#include "detection_layer.h"
#include "convolutional_layer.h"
#include "maxpool_layer.h"
#include "dropout_layer.h"
#include "blas.h"
}

float * get_network_output_gpu_layer(network net, int i);
float * get_network_delta_gpu_layer(network net, int i);
float * get_network_output_gpu(network net);

void forward_network_gpu(network net, network_state state)
{
    state.workspace = net.workspace;
    int i;
    for(i = 0; i < net.n; ++i){
        state.index = i;
        layer l = net.layers[i];
        if(l.delta_gpu){
            fill_ongpu(l.outputs * l.batch, 0, l.delta_gpu, 1);
        }
        if(l.type == CONVOLUTIONAL){
            forward_convolutional_layer_gpu(l, state);
        } else if(l.type == DETECTION){
            forward_detection_layer_gpu(l, state);
        } else if(l.type == CONNECTED){
            forward_connected_layer_gpu(l, state);
        } else if(l.type == CROP){
            forward_crop_layer_gpu(l, state);
        }else if(l.type == MAXPOOL){
            forward_maxpool_layer_gpu(l, state);
        } else if(l.type == DROPOUT){
            forward_dropout_layer_gpu(l, state);
        } 
        state.input = l.output_gpu;
    }
}

void update_network_gpu(network net)
{
    int i;
    int update_batch = net.batch*net.subdivisions;
    float rate = get_current_rate(net);
    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        if(l.type == CONVOLUTIONAL){
            update_convolutional_layer_gpu(l, update_batch, rate, net.momentum, net.decay);
        } else if(l.type == CONNECTED){
            update_connected_layer_gpu(l, update_batch, rate, net.momentum, net.decay);
        }
    }
}

void forward_backward_network_gpu(network net, float *x, float *y)
{
    network_state state;
    state.index = 0;
    state.net = net;
    int x_size = get_network_input_size(net)*net.batch;
    int y_size = get_network_output_size(net)*net.batch;
    if(net.layers[net.n-1].truths) y_size = net.layers[net.n-1].truths*net.batch;
    if(!*net.input_gpu){
        *net.input_gpu = cuda_make_array(x, x_size);
        *net.truth_gpu = cuda_make_array(y, y_size);
    }else{
        cuda_push_array(*net.input_gpu, x, x_size);
        cuda_push_array(*net.truth_gpu, y, y_size);
    }
    state.input = *net.input_gpu;
    state.delta = 0;
    state.truth = *net.truth_gpu;
    state.train = 1;
    forward_network_gpu(net, state);
    //backward_network_gpu(net, state);
}


typedef struct {
    network net;
    float *X;
    float *y;
} train_args;


float *get_network_output_layer_gpu(network net, int i)
{
    layer l = net.layers[i];
    cuda_pull_array(l.output_gpu, l.output, l.outputs*l.batch);
    return l.output;
}

float *get_network_output_gpu(network net)
{
    int i;
    for(i = net.n-1; i > 0; --i) if(net.layers[i].type != COST) break;
    return get_network_output_layer_gpu(net, i);
}

float *network_predict_gpu(network net, float *input)
{
    int size = get_network_input_size(net) * net.batch;
    network_state state;
    state.index = 0;
    state.net = net;
    state.input = cuda_make_array(input, size);
    state.truth = 0;
    state.train = 0;
    state.delta = 0;
    forward_network_gpu(net, state);
    float *out = get_network_output_gpu(net);
    cuda_free(state.input);
    return out;
}

