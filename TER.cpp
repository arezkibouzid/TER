#include<opencv2/opencv.hpp>
#include<iostream>
#include<string>

#include "D:\workspace\TER\TER\cTracker2.h"
#include "D:\workspace\TER\TER\cBlob.h"
#include"math.h"


#define M_PI 3.14159265358979323846
using namespace std;
using namespace cv;



Mat capture_frame, filter_frame, balance_frame, gaussian_frame, threshold_frame;

bool balance_flag = false;
int kernel_size = 3;
int block_size = 3;
int c = 0;
double segma = 0;
string path_image = "C:/Users/Arezki Bouzid/Desktop/Sans titre.png";

void capture();
void filter();

Mat& filterFrame();
Mat& gaussianFrame();
Mat& thresholdFrame();
Mat classifier(cv::Mat& img, cTracker2& test, int codif);


Point GetCentroid(Mat& img) {

   

    Moments m = moments(img, true);
   return  Point(m.m10 / m.m00, m.m01 / m.m00);

   



}


double ManhattanDistance(vector<double>& a, vector<double>& b) {

    double dist = 0;
    int i;
    for (i = 0; i < min(a.size(),b.size()); i++)
    {
        dist += abs(a.at(i) - b.at(i));

    }
    int j = i;
    while (i < a.size())
    {
        dist += abs(a.at(i));
        i++;
    }

    while (j < b.size())
    {
        dist += abs(b.at(j));
        j++;
    }

    return dist;

}

vector<double> GFD(Mat& img, int m ,int n) {


    Point p = GetCentroid(img);

    // a verifier 
    double maxRad = sqrt(std::pow(p.x, 2) + std::pow(p.y, 2));


    cout << "width" << img.size().width << endl;
    cout << "height" << img.size().height << endl;

    cout << p << endl;

    double radius, tempR, tempI;
    double theta;

    
    vector<vector<double>> FR;
    FR.resize(m, vector<double>(n));
    vector<vector<double>> FI;
    FI.resize(m, vector<double>(n));


    for (int rad = 0; rad < m; rad++)
    {

        for (int ang = 0; ang < n; ang++)
        {

            for (int x = p.x; x < img.size().width; x++)
            {       
               
                for (int y = p.y; y < img.size().height; y++)
                {

                    
                    radius = sqrt(std::pow(x - p.x, 2) + std::pow(y - p.y, 2));
                    theta = atan2((x - p.x), (y - p.y));
                    if (theta < 0) theta += 2 * M_PI;

                    
                    cout << to_string(img.at<uchar>(Point(x, y)));
                    tempR = img.at<uchar>(Point(x,y))*std::cos(2 * M_PI * rad * (radius /  maxRad) + ang * theta);
                    tempI = img.at<uchar>(Point(x,y))*std::sin(2 * M_PI * rad * (radius / maxRad) + ang * theta);

                
                    FR.at(rad).at(ang)+= tempR;
                    FI.at(rad).at(ang)-= tempI;

                  



                }
            }

           
               
        }
    }

 
    

    int taille = (m) * (n);
    vector<double> FD(taille);
    double DC;

    for (int rad = 0; rad < m; rad++)
    {
        for (int ang = 0; ang < n; ang++)
        {
            if (rad == 0 && ang == 0) {
                
                DC = sqrt(std::pow(FR.at(0).at(0),2) + std::pow(FR.at(0).at(0),2));
                
                FD.at(0)= DC /( M_PI * std::pow(maxRad,2));
            }
            else {
                int pos = rad * n + ang;
                FD.at(pos) = sqrt(std::pow(FR.at(rad).at(ang),2) + std::pow(FI.at(rad).at(ang),2)) / DC;
            }
               
        }
    }

    return FD;


}
int main()
{
    capture();

    filter();


    namedWindow("threshold_frame", WINDOW_AUTOSIZE);
    imshow("threshold_frame", threshold_frame);

    vector<double> vc= GFD(threshold_frame, 3, 50);

    for (int i = 0; i < vc.size(); i++) {
        std::cout << vc.at(i) << ' ';
    }

  
    

    //cTracker2 test = cTracker2(0, 1);


    //test.extractBlobs(filter_frame);






    //cout << test.zonesCount() << endl;

    



    //namedWindow("res", WINDOW_AUTOSIZE);
    //imshow("res", classifier(threshold_frame, test, 0));
    //namedWindow("res1", WINDOW_AUTOSIZE);
    //imshow("res1", classifier(threshold_frame, test, 100));











    

    waitKey(0);
    return 0;
}




cv::Mat classifier(cv::Mat& img, cTracker2& test, int codif) {




    Mat res;
    int height = test.getBlobs()[codif].max.y;
    int longeur = img.size().width;
    int width = test.getBlobs()[codif].max.x;
    int minY = test.getBlobs()[codif].min.y;
    int minX = test.getBlobs()[codif].min.x;
    int maxY = test.getBlobs()[codif].max.y;
    int maxX = test.getBlobs()[codif].max.x;

    //+2 pour le decalager
    res = cv::Mat(width - minX, height - minY, CV_8UC1);

    for (int x = minX; x < maxX; ++x)
    {
        uchar* row_ptr = res.ptr<uchar>(x - minX);


        for (int y = minY; y < maxY; ++y)
        {
            if (test.getlabels()[y * (longeur)+x]->i == codif) {

                //+2 pour le decalager
                row_ptr[y - minY] = 0;
                //if (x == minX && y == minY) row_ptr[y-minY] = 0;
                //if (x == maxX && y == maxY) row_ptr[y-minY] = 0;

                //if (x == test.getBlobs()[codif].location.x && y == test.getBlobs()[codif].location.y) row_ptr[y-minY] = 0;
            }
            else  row_ptr[y - minY] = 255;





        }
    }



    res = res.t();
    return res;



}

void capture() {

    capture_frame = imread(path_image);

}


// capture frame, convert to grayscale, apply Gaussian blur, apply balance (if applicable), and apply adaptive threshold method
void filter() {


    cvtColor(capture_frame, filter_frame, COLOR_BGR2GRAY);
    GaussianBlur(filter_frame, gaussian_frame, cv::Size(kernel_size, kernel_size), segma, segma);
    if (balance_flag) absdiff(gaussian_frame, balance_frame, gaussian_frame);
    threshold(gaussian_frame, threshold_frame, 10, 255, cv::THRESH_BINARY);
    //imwrite("./threshold_frame.jpg", threshold_frame);

}



cv::Mat& filterFrame() {
    return filter_frame;
}

cv::Mat& gaussianFrame() {
    return gaussian_frame;
}

cv::Mat& thresholdFrame() {
    return threshold_frame;
}