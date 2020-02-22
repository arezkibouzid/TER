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
Mat capture_frame2, filter_frame2, balance_frame2, gaussian_frame2, threshold_frame2;

bool balance_flag = false;
int kernel_size = 3;
int block_size = 3;
int c = 0;
double segma = 0;
string path_image = "C:/Users/Arezki Bouzid/Desktop/plz.png";
string path_image2 = "C:/Users/Arezki Bouzid/Desktop/plz - Copie (2).png";

void capture(Mat& capture_frame,string path);
void filter(Mat& capture_frame,Mat& threshold_frame);

Mat& filterFrame();
Mat& gaussianFrame();
Mat& thresholdFrame();
Mat classifier(cv::Mat& img, cTracker2& test, int codif);


Point GetCentroid(Mat& img) {

    Moments m = moments(img, true);
   
   return  Point(m.m10 / m.m00, m.m01 / m.m00);

}

vector<Point> GetExtrema(Mat& img) {


    // Find contours
    vector<vector<Point>> cnts;
  
    cv::findContours(img, cnts,RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    //cnts = imutils.grab_contours(cnts);
    //auto c = std::max(cnts, contourArea);
    if (cnts.size() > 1) cout << "plusieurs contours";

    int min_x = cnts.at(0).at(0).x;
    int max_x = min_x;

    int min_y = cnts.at(0).at(0).y;
    int max_y = min_y;
    Point left;
    Point top;
    Point right;
    Point bottom;

 
    for (Point p : cnts.at(0))
    {
        
        if (min_x > p.x) { left = p;  min_x = p.x;  }
        if (min_y > p.y) { top = p;  min_y = p.y; }
        if (max_x < p.x) { right = p;  max_x = p.x; }
        if (max_y < p.y) { bottom = p;  max_y = p.y; }
    }
    vector<Point> extrema;
    extrema.push_back(left);
    extrema.push_back(top);
    extrema.push_back(right);
    extrema.push_back(bottom);

    return extrema;

        
}

double distance(Point p1, Point p2)
{
    double res;
    res = sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p1.y));
    return res;
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


    Point Centroid = GetCentroid(img);


    vector<Point> extrema = GetExtrema(img);

    // a verifier 
    double maxRad = 0;

    for (Point p : extrema) {

        maxRad = (maxRad < distance(Centroid,p)) ? distance(Centroid,p) : maxRad;


    }
   


    cout << "width" << img.size().width << endl;
    cout << "height" << img.size().height << endl;

    cout << Centroid << endl;

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
            // a verifier  x and y
            for (int x = 0; x < img.size().width; x++)
            {       
               
                for (int y = 0; y < img.size().height; y++)
                {

                    
                    radius = sqrt(std::pow(x - Centroid.x, 2) + std::pow(y - Centroid.y, 2));
                    theta = atan2((y - Centroid.y), (x - Centroid.x));
                    if (theta < 0) theta += 2 * M_PI;

                    
                    //cout << to_string(img.at<uchar>(Point(x, y)));
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



void circshift(Mat& out, const Point& delta)
{
    Size sz = out.size();

    
    assert(sz.height > 0 && sz.width > 0);
  

    if ((sz.height == 1 && sz.width == 1) || (delta.x == 0 && delta.y == 0))
        return;

  
    int x = delta.x;
    int y = delta.y;
    if (x > 0) x = x % sz.width;
    if (y > 0) y = y % sz.height;
    if (x < 0) x = x % sz.width + sz.width;
    if (y < 0) y = y % sz.height + sz.height;


    
    vector<Mat> planes;
    split(out, planes);

    for (size_t i = 0; i < planes.size(); i++)
    {
        
        Mat tmp0, tmp1, tmp2, tmp3;
        Mat q0(planes[i], Rect(0, 0, sz.width, sz.height - y));
        Mat q1(planes[i], Rect(0, sz.height - y, sz.width, y));
        q0.copyTo(tmp0);
        q1.copyTo(tmp1);
        tmp0.copyTo(planes[i](Rect(0, y, sz.width, sz.height - y)));
        tmp1.copyTo(planes[i](Rect(0, 0, sz.width, y)));

       
        Mat q2(planes[i], Rect(0, 0, sz.width - x, sz.height));
        Mat q3(planes[i], Rect(sz.width - x, 0, x, sz.height));
        q2.copyTo(tmp2);
        q3.copyTo(tmp3);
        tmp2.copyTo(planes[i](Rect(x, 0, sz.width - x, sz.height)));
        tmp3.copyTo(planes[i](Rect(0, 0, x, sz.height)));
    }

    merge(planes, out);
}

Mat centreObject(Mat& img) {

    int width = img.size().width;
    int height = img.size().height;
    auto type = img.type();
    


    Mat ligne = cv::Mat::zeros(1, width, type);
    Mat colonne = cv::Mat::zeros(height,1, type);
    

    Mat dist=img;

    int temp = std::max(width, height)-std::min(width, height) * 0.5;
    
    if (height < width) {
        
       
        if ((temp % 1) > 0) {
            hconcat(dist, colonne, dist);
        }

        for (int i = 0; i < round(temp) ; i++)
        {
            dist.push_back(ligne);
        }
       
        
    }
    else if (width < height) {
        

        if ((temp % 1) > 0) dist.push_back(ligne);
        
        for (int i = 0; i < round(temp); i++)
        {
            hconcat(dist,colonne, dist);
        }
        
    }

    
   
       Point state = GetCentroid(dist);

      
        width = dist.size().width;
        height = dist.size().height;
    
    int delta_y = round(height / 2 - state.y);
    int delta_x = round(width / 2 - state.x);
    int delta_max = max(abs(delta_y), abs(delta_x));
       
    colonne = cv::Mat::zeros(height, 1, type);
    for (int i = 0; i < delta_max + 10; i++) {
        hconcat(dist, colonne, dist);
        hconcat(colonne, dist, dist);
    }

   
     ligne = cv::Mat::zeros(1, dist.size().width, type);

   
     for (int i = 0; i < delta_max + 10; i++) {
         vconcat(ligne, dist, dist);
         dist.push_back(ligne);
     }


        circshift(dist, Point(delta_x, delta_y));




    return dist;


}



int main()
{

  
    capture(capture_frame,path_image);
    filter(capture_frame,threshold_frame);

    capture(capture_frame2,path_image2);
    filter(capture_frame2,threshold_frame2);
    



    namedWindow("threshold_frame", WINDOW_NORMAL);
    imshow("threshold_frame", threshold_frame);

    namedWindow("threshold_frame2", WINDOW_AUTOSIZE);
    imshow("threshold_frame2", threshold_frame2);

    
  
    Mat dist = centreObject(threshold_frame);
    Mat dist2 = centreObject(threshold_frame2);

   


    namedWindow("centreObject", WINDOW_NORMAL);
    imshow("centreObject", dist);
    namedWindow("centreObject2", WINDOW_NORMAL);
    imshow("centreObject2", dist2);

 
 


    vector<double> vc= GFD(threshold_frame, 10, 10);
    vector<double> vc2 = GFD(threshold_frame2, 10, 10);



  double diste = ManhattanDistance(vc, vc2);

    cout << diste << endl;
 //   for (int i = 0; i < vc.size(); i++) {
 //       std::cout << vc.at(i) << ' ';
  //  }

 
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

void capture(Mat& capture_frame,string path) {

    capture_frame = imread(path);

}



void filter(Mat& capture_frame, Mat& threshold_frame) {

    // capture frame, convert to grayscale, apply Gaussian blur, apply balance (if applicable), and apply adaptive threshold method
    cvtColor(capture_frame, capture_frame, COLOR_BGR2GRAY);
    //GaussianBlur(filter_frame, gaussian_frame, cv::Size(kernel_size, kernel_size), segma, segma);
   // if (balance_flag) absdiff(gaussian_frame, balance_frame, gaussian_frame);
    threshold(capture_frame, threshold_frame, 10, 255, cv::THRESH_BINARY);
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