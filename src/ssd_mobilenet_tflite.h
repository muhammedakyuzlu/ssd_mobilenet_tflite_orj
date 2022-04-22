#pragma once

#include <cstdint>
#include <vector>
#include <chrono>
#include <iostream>
#include <fstream>
#include <cmath>

#include <tensorflow/lite/model.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>

// #include <model.h>
// #include <interpreter.h>
// #include <kernels/register.h>



#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

struct Prediction
{
    std::vector<cv::Rect> boxes;
    std::vector<float>    scores;
    std::vector<int>      labels;
};



class SSD_MOBILENET
{
public:
    // Take a model path as string
    void loadModel(const std::string path);
    // Take an image and return a prediction
    void run(cv::Mat image, Prediction &out_pred);

    void getLabelsName(std::string path, std::vector<std::string> &labelNames);

    // thresh hold
    float confThreshold = 0.1;
    float nmsThreshold = 0.1;

    // number of threads
    int nthreads = 4;

    int boxes_idx = 0; 
    int classes_idx = 1;
    int scores_idx = 2;
    int num_idx = 3;

private:
    // model's
    std::unique_ptr<tflite::FlatBufferModel> _model;
    std::unique_ptr<tflite::Interpreter> _interpreter;
    tflite::StderrReporter _error_reporter;

    // parameters of interpreter's input
    int _input;
    int _in_height;
    int _in_width;

    // // parameters of interpreter's output
    TfLiteTensor* output_locations = nullptr;
    TfLiteTensor* output_classes = nullptr;
    TfLiteTensor* num_detections = nullptr;
    TfLiteTensor* output_scores = nullptr;

    // Input of the interpreter
    uint8_t *_input_8;
    float  *_input_32;

    void preprocess(cv::Mat &image);
};