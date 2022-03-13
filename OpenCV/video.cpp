#include<iostream>
#include<opencv2/opencv.hpp>
#include<vector>
#include<string>

using namespace std;
using namespace cv;

int main(int argc, char *argv[]){

    VideoCapture cap(argv[1]);
    VideoWriter videoWriter;

    if(!cap.isOpened()){
        cout<<"Can't open video"<<endl;
        return -1;
    }
    float fps = cap.get(CAP_PROP_FPS);
    int width = cap.get(CAP_PROP_FRAME_WIDTH);
    int height = cap.get(CAP_PROP_FRAME_HEIGHT);
    cout << "FPS: "<< fps<<" ,frame width: "<<width<<" ,frame height: "<<height<<endl;
    string output_path = "./result/another_videop.avi";
    videoWriter.open(output_path, VideoWriter::fourcc('M','J','P','G'), fps, Size(width, height), true);


    Mat frame;
    while(1){
        cap >> frame;
        if(frame.empty()){
            cout<<"empty frame"<<endl;
            return 0;
        }
        videoWriter<<frame;
        imshow("camera img", frame);
        if(waitKey(25) == 27){break;}
    }
    return 0;
}
