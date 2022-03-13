#include<iostream>
#include<vector>
#include<opencv2/opencv.hpp>
using namespace std;
using namespace cv;

int main(int argc, char *argv[]){

    if(argc!=2){
        cerr << "./file img_dir"<<endl;
        return -1;
    }

    Mat img;
    img = imread(argv[1]);
    
    if (img.empty()) {
        cerr << "Image load failed!" << endl;
        return -1;
    }
    

    Mat img_roi_1, img_roi_2;
    img_roi_1 = img(Rect(200,200,100,100));
    img_roi_2 = img(Rect(Point(300,200), Point(400,300)));
    imshow("image", img);
    imshow("img_Roi_1", img_roi_1);
    imshow("img_Roi_2", img_roi_2);
    waitKey(0);

    return 0;
}
