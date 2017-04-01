#include "network.h"
#include "detection_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "demo.h"
#include <sys/time.h>

#define FRAMES 3

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
void convert_detections(float *predictions, int classes, int num, int square, int side, int w, int h, float thresh, float **probs, box *boxes, int only_objectness);

/* Comment on the overral code logic
 * basically, after geting a frame from the stream, namly in, it is first resized to in_s
 * this procedure is done using the fetch function
 * in the detection funciton, work are done on det and det_s which essentially point to teh
 * same image in and in_s represent. 
 * in the detect function, it maintain an internal buffer, which has the length of FRAME.
 * this means all the work on a single frame is relevant on the adjacent FRAME frames
 * so, in the detection function first the original image det and resized det_s are passed in
 * (the reason det is needed is for printing out again)
 * the inference made on the det_s in stored in prediction, a 1-d array
 * prediction are placed in 2-d array predictions(which has a depth of FRAME), then every final prediction 
 * are draw from all the FRAME predictions by taking an average. 
 * then by using det, the original frame is put into the internal buffer of images,
 * some other routines process the pictures in images and put box on them
 * then the pointer det is used again to refer to the picture-processed on the last frame and return back
 * finally the demo got this processed image by refering to det and put it on the output stream
 */

static char **demo_names;
static image *demo_labels;
static int demo_classes;

static float **probs;
static box *boxes;
static network net;
static image in   ;
static image in_s ;
static image det  ;
static image det_s;
static image disp = {0};
static CvCapture * cap;
static float fps = 0;
static float demo_thresh = 0;

static float *predictions[FRAMES];
static int demo_index = 0;
static image images[FRAMES];
static float *avg;

void *fetch_in_thread(void *ptr)
{
    in = get_image_from_stream(cap);
    if(!in.data){
        error("Stream closed.");
    }
    in_s = resize_image(in, net.w, net.h);
    return 0;
}

void *detect_in_thread(void *ptr)
{
    float nms = .1;

    detection_layer l = net.layers[net.n-1];
    float *X = det_s.data;
    float *prediction = network_predict(net, X);

    memcpy(predictions[demo_index], prediction, l.outputs*sizeof(float));
    //defined in util.h, average each col of a 2-d matrix, output 1-d array
    //squeeze on the 2nd paramenter dimension
    mean_arrays(predictions, FRAMES, l.outputs, avg);

    free_image(det_s);
    convert_detections(avg, l.classes, l.n, l.sqrt, l.side, 1, 1, demo_thresh, probs, boxes, 0);
    if (nms > 0) do_nms(boxes, probs, l.side*l.side*l.n, l.classes, nms);
    printf("\033[2J");
    printf("\033[1;1H");
    printf("\nFPS:%.1f\n",fps);
    printf("Objects:\n\n");

    images[demo_index] = det;
    det = images[(demo_index + FRAMES/2 + 1)%FRAMES]; //point to a frame before, in a loop fashion
    demo_index = (demo_index + 1)%FRAMES; //point to a frame after

    draw_detections(det, l.side*l.side*l.n, demo_thresh, boxes, probs, demo_names, demo_labels, demo_classes);

    return 0;
}

double get_wall_time()
{
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, image *labels, int classes, int frame_skip)
{
    int delay = frame_skip;
    demo_names = names;
    demo_labels = labels;
    demo_classes = classes;
    demo_thresh = thresh;
    printf("Demo\n");
    net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);

    srand(2222222);

    if(filename){
        cap = cvCaptureFromFile(filename);
    }else{
        cap = cvCaptureFromCAM(cam_index);
    }

    if(!cap) error("Couldn't connect to webcam.\n");

    detection_layer l = net.layers[net.n-1];
    int j;

    avg = (float *) calloc(l.outputs, sizeof(float));
    for(j = 0; j < FRAMES; ++j) predictions[j] = (float *) calloc(l.outputs, sizeof(float));
    for(j = 0; j < FRAMES; ++j) images[j] = make_image(1,1,3);

    boxes = (box *)calloc(l.side*l.side*l.n, sizeof(box));
    probs = (float **)calloc(l.side*l.side*l.n, sizeof(float *));
    for(j = 0; j < l.side*l.side*l.n; ++j) probs[j] = (float *)calloc(l.classes, sizeof(float *));

    pthread_t fetch_thread;
    pthread_t detect_thread;

    //the reasons for below operations is we need to pack the buffers before the iterations can smoothly start
    fetch_in_thread(0); // in and in_s will get initial value after this step
    det = in; //notice, every time we need to pass in and in_s to det and det_s
    det_s = in_s;

    fetch_in_thread(0);
    detect_in_thread(0);
    disp = det;
    det = in;
    det_s = in_s;

    for(j = 0; j < FRAMES/2; ++j){
        fetch_in_thread(0);
        detect_in_thread(0);
        disp = det;
        det = in;
        det_s = in_s;
    }

    //main opeariton iteration
    int count = 0;
    cvNamedWindow("Demo", CV_WINDOW_NORMAL); 
    cvMoveWindow("Demo", 0, 0);
    //cvResizeWindow("Demo", 1352, 1013);
    cvResizeWindow("Demo", 448,448);

    double before = get_wall_time();

    while(1){
        ++count;
        if(1){
            if(pthread_create(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed");
            if(pthread_create(&detect_thread, 0, detect_in_thread, 0)) error("Thread creation failed");

            if(1){
                show_image(disp, "Demo");
                int c = cvWaitKey(1);
                if (c == 10){
                    if(frame_skip == 0) frame_skip = 60;
                    else if(frame_skip == 4) frame_skip = 0;
                    else if(frame_skip == 60) frame_skip = 4;   
                    else frame_skip = 0;
                }
            }else{
                char buff[256];
                sprintf(buff, "/home/pjreddie/tmp/bag_%07d", count);
                save_image(disp, buff);
            }

            pthread_join(fetch_thread, 0);
            pthread_join(detect_thread, 0);

            if(delay == 0){
                free_image(disp);
                disp  = det;
            }
            det   = in;
            det_s = in_s;
        }else {
            fetch_in_thread(0);
            det   = in;
            det_s = in_s;
            detect_in_thread(0);
            if(delay == 0) {
                free_image(disp);
                disp = det;
            }
            show_image(disp, "Demo");
            cvWaitKey(1);
        }
        --delay;
        if(delay < 0){
            delay = frame_skip;

            double after = get_wall_time();
            float curr = 1./(after - before);
            fps = curr;
            before = after;
        }
    }
}
#else
void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, image *labels, int classes, int frame_skip)
{
    fprintf(stderr, "Demo needs OpenCV for webcam images.\n");
}
#endif

