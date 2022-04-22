#include "ssd_mobilenet_tflite.h"

void SSD_MOBILENET::getLabelsName(std::string path, std::vector<std::string> &labelNames)
{
    // Open the File
    std::ifstream in(path.c_str());
    // Check if object is valid
    if (!in)
        throw std::runtime_error("Can't open ");
    std::string str;
    // Read the next line from File until it reaches the end.
    while (std::getline(in, str))
    {
        // Line contains string of length > 0 then save it in vector
        if (str.size() > 0)
            labelNames.push_back(str);
    }
    // Close The File
    in.close();
}

void SSD_MOBILENET::loadModel(const  std::string path)
{
     // const char * c = str.c_str();   
    _model = tflite::FlatBufferModel::BuildFromFile(path.c_str());
    if (!_model)
    {
        std::cout << "\nFailed to load the model.\n"
                  << std::endl;
        exit(1);
    }
    else
    {
        std::cout << "\nModel loaded successfully.\n";
    }
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*_model, resolver)(&_interpreter);
    TfLiteStatus status = _interpreter->AllocateTensors();
    if (status != kTfLiteOk)
    {
        std::cout << "\nFailed to allocate the memory for tensors.\n"
                  << std::endl;
        exit(1);
    }
    else
    {
        std::cout << "\nMemory allocated for tensors.\n";
    }

    // input information
    _input = _interpreter->inputs()[0];
    TfLiteIntArray *dims = _interpreter->tensor(_input)->dims;
    _in_height = dims->data[1];
    _in_width = dims->data[2];
    _input_8 = _interpreter->typed_tensor<uint8_t>(_input);
    _interpreter->SetNumThreads(nthreads);
     

}

void SSD_MOBILENET::preprocess(cv::Mat &image)
{
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    cv::resize(image, image, cv::Size(_in_height, _in_width), cv::INTER_CUBIC);
    image.convertTo(image, CV_8U);

}

void SSD_MOBILENET::run(cv::Mat frame, Prediction &out_pred)
{
    cv::Size s = frame.size();
    preprocess(frame);
    memcpy(_input_8, frame.data, frame.total() * frame.elemSize());
    _interpreter->SetAllowFp16PrecisionForFp32(true);

    // Inference
    TfLiteStatus status = _interpreter->Invoke();
    if (status != kTfLiteOk)
    {
        std::cout << "\nFailed to run inference!!\n";
        exit(1);
    }




    // bbox
    output_locations = _interpreter->tensor(_interpreter->outputs()[boxes_idx]);
    auto out_location = output_locations->data.f;
    
    // label/class
    output_classes   = _interpreter->tensor(_interpreter->outputs()[classes_idx]);
    auto out_cls = output_classes->data.f;
    
    // scores
    output_scores   = _interpreter->tensor(_interpreter->outputs()[scores_idx]);
    auto out_score = output_scores->data.f;

    // number of detection
    num_detections   = _interpreter->tensor(_interpreter->outputs()[num_idx]);
    auto nums_det = num_detections->data.f; 
    



    std::vector<float> locations;
    std::vector<float> cls;

    // convert the outputs from tensor to vectors
    for (int i = 0; i < 100; i++){
        locations.push_back(out_location[i]);
        cls.push_back(out_cls[i]);
    }

    int count=0;
    int ymin, xmin, ymax,  xmax;
    std::vector<cv::Rect> boxes;
    std::vector<float>    scores;
    std::vector<int>      classIDs;
    
    for(int j = 0; j < *nums_det*4; j+=4){

        ymin = locations[j+0] * s.height;
        xmin = locations[j+1] * s.width;
        ymax = locations[j+2] * s.height;
        xmax = locations[j+3] * s.width;

        auto width  = xmax - xmin;
        auto height = ymax - ymin;

        float score = out_score[count];

        boxes.push_back(cv::Rect(xmin, ymin, width, height));
        scores.push_back(score);
        classIDs.push_back(cls[count]);

        count+=1;
    }
    std::vector<int> indices;
 
    cv::dnn::NMSBoxes(boxes, scores, confThreshold, nmsThreshold, indices);
    
    for (int i = 0; i < indices.size(); i++)
    {
        out_pred.boxes.push_back(boxes[indices[i]]);
        out_pred.scores.push_back(scores[indices[i]]);
        out_pred.labels.push_back(classIDs[indices[i]]);
    }
};
