#include<iostream>
#include<vector>
#include<opencv2/opencv.hpp>
#include<chrono>
#include<string>
#include<fstream>
#include<sstream>
#include<unordered_set>

using namespace std;
using namespace cv;
using namespace dnn;

float confThreshold = 0.5;
float maskThreshold = 0.3;

vector<string> classes;
vector<Scalar> colors;
unordered_set<string> dynamicClasses;

string classfile = "./pretrained/mscoco_labels.names";
string colorfile = "./pretrained/colors.txt";
string dynamicfile = "./pretrined/dynamic_object.txt";
string textGraph = "./pretrained/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt";
string modelWeights = "./pretrained/mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb";

void txt_process();
void postprocess(Mat& frame, const vector<Mat>& outs, Mat& dynamic_mask);
void drawBox(Mat& frame, int classId, float score, Rect box);
void drawMask(Mat& frame, int classId, Rect box, Mat &mask);
bool is_dynamic(int classId);


int main(int argc, char *argv[]){

    if(argc != 2){
        cerr << "./build_file image_path"<<endl;
        exit(0);
        return -1;

    }

    cout << "Mask RCNN inference start" <<endl;
    txt_process();

    // Load model
    Net net = readNetFromTensorflow(modelWeights, textGraph);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    cout << "model weight upload" << endl;

    string str, outputFile;
    outputFile = "./result/MaskRcnn.png";
    VideoCapture cap;
    VideoWriter video;
    Mat frame, blob;
    
    //cap.open("./data/john_wick01.jpg");
    cap.open(argv[1]);

    static const string kWinName = "Mask R-CNN in OpenCV";
    namedWindow(kWinName, WINDOW_NORMAL);

    cap >> frame;


    if (frame.empty())
    {
        cout << "Done processing !!!" << endl;
        cout << "Output file is stored as " << outputFile << endl;
        waitKey(3000);
    }

    
    blobFromImage(frame, blob, 1.0, Size(frame.cols, frame.rows), Scalar(), true, false);   
    cout << "blob size = " << blob.size << endl;

    net.setInput(blob);
    std::vector<String> outNames(2);
    outNames[0] = "detection_out_final";
    outNames[1] = "detection_masks";
    vector<Mat> outs;

    cout<< "network forwarding start" << endl;
    // std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    net.forward(outs, outNames);
    // std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    // double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
    // std::cout << "network forwarding time " << ttrack << std::endl;

    Mat dynamicMask;

    cout << "postprocess start "<<endl;
    postprocess(frame, outs, dynamicMask);
    cout<< "postprocess end" <<endl;

    vector<double> layersTimes;
    double freq = getTickFrequency() / 1000;
    double t = net.getPerfProfile(layersTimes) / freq;
    string label = format("Mask-RCNN on 3.6 GHz Intel Core i7 CPU, Inference time for a frame : %0.0f ms", t);
    putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));

    cout << "result image storing"<<endl;
    
    imwrite("./result/dynamic_mask.png", dynamicMask);
    imwrite(outputFile, frame);
    imshow(kWinName, frame);
    imshow("dynamic_mask", dynamicMask);

    cap.release();
    cout << "all processing end"<<endl;
    return 0;
}

void txt_process(){

    ifstream ifs(classfile.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);

    cout << "classfile upload" <<endl;

    ifstream ifs2(dynamicfile.c_str());
    while(getline(ifs2,line)){
        dynamicClasses.insert(line);
    }

    cout << "dynamic object upload" << endl;

    ifstream colorFptr(colorfile.c_str());
    while(getline(colorFptr, line)){
        char* pEnd;
        double r,g,b;
        r = strtod(line.c_str(), &pEnd);
        g = strtod(pEnd, NULL);
        b = strtod(pEnd, NULL);
        colors.push_back(Scalar(r,g,b, 255.0));
    }

    cout << "color file upload"<<endl;
}


void postprocess(Mat& frame, const vector<Mat>& outs, Mat& dynamic_mask){
    Mat outDetections = outs[0];
    Mat outMask = outs[1];

    // outDetection : (1,1,numDetection, box_info)
    // outMask: (numDetection, numClasses, msk_width, msk_height)
    cout << "out detection size" << outDetections.size()<<endl;
    cout << "out maks size" << outMask.size()<<endl;

    const int numdetection = outDetections.size[2];
    const int numClasses = outMask.size[1];

    // _, classId, score, left, top, right, bottom
    outDetections = outDetections.reshape(1, outDetections.total() / 7);
    cout << "out Detection after resize = " << outDetections.size << endl;   

    dynamic_mask = Mat(frame.size(), CV_8U, Scalar(255));
    Mat mat_zeros = Mat::zeros(frame.size(), CV_8U);

    for(int i = 0 ; i<numdetection; i++){
        float score = outDetections.at<float>(i,2);
        if(score> confThreshold){

            int classId = static_cast<int>(outDetections.at<float>(i,1));
            int left = static_cast<int>(frame.cols * outDetections.at<float>(i,3));
            int top =  static_cast<int>(frame.rows * outDetections.at<float>(i,4));
            int right =  static_cast<int>(frame.cols * outDetections.at<float>(i,5));
            int bottom =  static_cast<int>(frame.rows * outDetections.at<float>(i,6));
            

            left = max(0, min(left, frame.cols - 1));
            top = max(0, min(top, frame.rows - 1 ));
            right = max(0, min(right, frame.cols -1));
            bottom = max(0, min(bottom, frame.rows -1));

            Rect box = Rect(left,top,right-left+1, bottom-top+1);

            Mat objectMask(outMask.size[2], outMask.size[3], CV_32F, outMask.ptr<float>(i,classId));
            resize(objectMask, objectMask, Size(box.width, box.height));
            Mat mask = (objectMask > maskThreshold);
            mask.convertTo(mask, CV_8U);

            drawBox(frame, classId, score, box); 
            drawMask(frame, classId, box, mask);

            if(is_dynamic(classId)){
                mat_zeros(box).copyTo(dynamic_mask(box), mask);
            }
        }
    }
}

void drawBox(Mat& frame, int classId, float score, Rect box){

    rectangle(frame, Point(box.x, box.y), Point(box.x+box.width, box.y+box.height), Scalar(255,178,50), 3);
    string label = format("%.2f", score);
    if(!classes.empty()){
        CV_Assert(classId< (int)classes.size());
        label = classes[classId] + ":" + label;
    }

    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5 ,1 , &baseLine);
    box.y = max(box.y, labelSize.height);
    rectangle(frame, Point(box.x, box.y - round(1.5*labelSize.height)), Point(box.x + round(1.5*labelSize.width), box.y+baseLine), Scalar(255,255,255), FILLED);
    putText(frame, label, Point(box.x, box.y), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,0),1);
}

void drawMask(Mat& frame, int classId, Rect box, Mat &mask){
    Scalar color = colors[classId%colors.size()];

    Mat colorRoi = Mat(Size(box.width, box.height), frame.type(), color);
    Mat colorMask = Mat::zeros(Size(box.width, box.height), frame.type());
    colorRoi.copyTo(colorMask, mask);
    
    addWeighted(frame(box), 0.5, colorMask, 0.5, 0.0, frame(box));

}

bool is_dynamic(int ClassId){
    return dynamicClasses.count(classes[ClassId]);
}

