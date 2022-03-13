#include<iostream>
#include<vector>
#include<opencv2/opencv.hpp>
using namespace std;


int main(int argc, char *argv[]){

    if(argc!=2){
        cerr << "./file img_dir"<<endl;
        return -1;
    }

    cv::Mat img;
    img = cv::imread(argv[1]);
    
    if (img.empty()) {
        cerr << "Image load failed!" << endl;
        return -1;
    }
    
    cv::imshow("image", img);
    cv::waitKey(0);

    return 0;
}
