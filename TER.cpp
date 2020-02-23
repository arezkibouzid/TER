#include<opencv2/opencv.hpp>
#include<iostream>
#include<string>
#include"math.h"

#include "CC.h"


#define M_PI 3.14159265358979323846


using namespace std;
using namespace cv;



Mat capture_frame, filter_frame, balance_frame, gaussian_frame, threshold_frame,centredImage;
Mat capture_frame2, filter_frame2, balance_frame2, gaussian_frame2, threshold_frame2, centredImage2;

Mat Matlabled;
vector<CC> composants;

vector<double> vectC;

bool balance_flag = false;
int kernel_size = 3;
int block_size = 3;
int c = 0;
double segma = 0;
string path_image = "plz.png";
string path_image2 = "plz - Copie (2).PNG";

void capture(Mat& capture_frame,string path);
void filter(Mat& capture_frame,Mat& threshold_frame);

Mat& filterFrame();
Mat& gaussianFrame();
Mat& thresholdFrame();


void connectedComponentsVector(Mat& threshold_frame,vector<CC>& composants);

double distance(Point& p1, Point& p2)
{
    double res;
    res = sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p1.y));
    return res;
}

double ManhattanDistance(vector<double>& a, vector<double>& b) {

    double dist = 0;
    int i;
    for (i = 0; i < min(a.size(), b.size()); i++)
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

Point GetCentroid(Mat& img) {

    Moments m = moments(img, true);
   
   return  Point(m.m10 / m.m00, m.m01 / m.m00);

}

double GetExtrema(Mat& img, Point& center) {


    // Find contours
    vector<vector<Point>> cnts;
  
    cv::findContours(img, cnts,RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    //cnts = imutils.grab_contours(cnts);
    //auto c = std::max(cnts, contourArea);
    if (cnts.size() > 1) cout << "plusieurs contours";

    /*int min_x = cnts.at(0).at(0).x;
    int max_x = min_x;

    int min_y = cnts.at(0).at(0).y;
    int max_y = min_y;
    Point lefttop;
    Point leftbottom;
    Point rightbottom;
    Point righttop;


    for (Point p : cnts.at(0))
    {
        
        if (min_x >= p.x && min_y >= p.y) { lefttop = p;  min_x = p.x;  min_y = p.y; }
        if (min_x >= p.x && min_y < p.y) { leftbottom = p;  min_y = p.y; min_x = p.x;}
        if (max_x <= p.x && max_y <= p.y) { rightbottom = p;  max_x = p.x;  max_y = p.y;}
        if (max_x <= p.x && max_y > p.y) { righttop = p;  max_y = p.y; max_x = p.x;}
    }
    cout << lefttop;
    cout << leftbottom;
    cout << rightbottom;
    cout << righttop;
    vector<Point> extrema;
    extrema.push_back(lefttop);
    extrema.push_back(leftbottom);
    extrema.push_back(rightbottom);
    extrema.push_back(righttop);
    */
 
    double maxRad = 0;
    for (Point p : cnts.at(0)) {

        if (maxRad < distance(center, p))  maxRad = distance(center, p);


    }

    return maxRad;

        
}



void centreObject(Mat& img, Mat& centredImage) {

    int width = img.size().width;
    int height = img.size().height;
    auto type = img.type();



    Mat ligne = cv::Mat::zeros(1, width, type);
    Mat colonne = cv::Mat::zeros(height, 1, type);


    centredImage = img;

    int temp = std::max(width, height) - std::min(width, height) * 0.5;

    if (height < width) {


        if ((temp % 1) > 0) {
            hconcat(centredImage, colonne, centredImage);
            ligne = cv::Mat::zeros(1, centredImage.size().width, type);
        }

        for (int i = 0; i < round(temp); i++)
        {
            centredImage.push_back(ligne);
        }


    }
    else if (width < height) {


        if ((temp % 1) > 0) {
            centredImage.push_back(ligne);
            colonne = cv::Mat::zeros(centredImage.size().height, 1, type);
        }
        for (int i = 0; i < round(temp); i++)
        {
            hconcat(centredImage, colonne, centredImage);
        }

    }



    Point state = GetCentroid(centredImage);


    width = centredImage.size().width;
    height = centredImage.size().height;

    int delta_y = round(height / 2 - state.y);
    int delta_x = round(width / 2 - state.x);
    int delta_max = max(abs(delta_y), abs(delta_x));

    colonne = cv::Mat::zeros(height, 1, type);
    for (int i = 0; i < delta_max + 10; i++) {
        hconcat(centredImage, colonne, centredImage);
        hconcat(colonne, centredImage, centredImage);
    }


    ligne = cv::Mat::zeros(1, centredImage.size().width, type);


    for (int i = 0; i < delta_max + 10; i++) {
        vconcat(ligne, centredImage, centredImage);
        centredImage.push_back(ligne);
    }


    circshift(centredImage, Point(delta_x, delta_y));





}


vector<double>& GFD(CC& composant, Mat& centredImage, int m ,int n) {



    centreObject(composant.getMat(), centredImage);
   
    
  
    cout << "width : " << centredImage.size().width << endl;
    cout << "height : " << centredImage.size().height << endl;

    Point Centroid = GetCentroid(centredImage);

    cout << Centroid << endl;
    
    double maxRad = GetExtrema(centredImage,Centroid);


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
            
            for (int x = 0; x < centredImage.size().width; x++)
            {       
               
                for (int y = 0; y < centredImage.size().height; y++)
                {

                    
                    radius = sqrt(std::pow(x - Centroid.x, 2) + std::pow(y - Centroid.y, 2));
                    theta = atan2((y - Centroid.y), (x - Centroid.x));
                    if (theta < 0) theta += 2 * M_PI;

                    
                    //cout << to_string(centredImage.at<uchar>(Point(x, y)));
                    tempR = centredImage.at<uchar>(Point(x,y))*std::cos(2 * M_PI * rad * (radius /  maxRad) + ang * theta);
                    tempI = centredImage.at<uchar>(Point(x,y))*std::sin(2 * M_PI * rad * (radius / maxRad) + ang * theta);

                
                    FR.at(rad).at(ang)+= tempR;
                    FI.at(rad).at(ang)-= tempI;

                  



                }
            }

           
               
        }
    }
    
   
 
    

    int taille = (m) * (n);
    vectC.resize(taille);
    double DC;

    for (int rad = 0; rad < m; rad++)
    {
        for (int ang = 0; ang < n; ang++)
        {
            if (rad == 0 && ang == 0) {
                
                DC = sqrt(std::pow(FR.at(0).at(0),2) + std::pow(FR.at(0).at(0),2));
                
                vectC.at(0)= DC /( M_PI * std::pow(maxRad,2));
            }
            else {
                int pos = rad * n + ang;
                vectC.at(pos) = sqrt(std::pow(FR.at(rad).at(ang),2) + std::pow(FI.at(rad).at(ang),2)) / DC;
            }
               
        }
    }

    return vectC;


}



int main()
{

  
    capture(capture_frame,path_image);
    filter(capture_frame,threshold_frame);




    //capture(capture_frame2, path_image2);
    //filter(capture_frame2, threshold_frame2);

    namedWindow("threshold_frame", WINDOW_NORMAL);
    imshow("threshold_frame", threshold_frame);

    //namedWindow("threshold_frame2", WINDOW_AUTOSIZE);
    //imshow("threshold_frame2", threshold_frame2);



    connectedComponentsVector(threshold_frame,composants);





    vector<vector<double>> vecteursCar;
    vector<double>& ptr_vect = GFD(composants.at(0), centredImage2, 3, 10);;

    namedWindow("composantCentred11", WINDOW_NORMAL);
    imshow("composantCentred11", centredImage2);

    for (int i = 1; i < composants.size(); i++)
    {
        
       ptr_vect=GFD(composants.at(i), centredImage, 3, 10);

       namedWindow("composantCentred22", WINDOW_NORMAL);
       imshow("composantCentred22",centredImage);

       vecteursCar.push_back(ptr_vect);
            
    }


    cout << composants.size();

    namedWindow("composants11", WINDOW_NORMAL);
    imshow("composants11", composants.at(0).getMat());
    namedWindow("composants22", WINDOW_NORMAL);
    imshow("composants22", composants.at(1).getMat());
  



  


   



    //double diste = ManhattanDistance(vc, vc2);

    //cout << diste << endl;

 //   for (int i = 0; i < vc.size(); i++) {
 //       std::cout << vc.at(i) << ' ';
  //  }

 
  


    

    waitKey(0);

    return 0;
 
}

void CcToMat(CC cc, Mat& img) {

    int width = cc.getdX();
    int height = cc.getdY();

    img = cv::Mat(width, height, CV_8UC1);

    int longeur = Matlabled.size().width;


    Point deb = cc.getPtr_debut();

    for (int x = deb.x ; x < deb.x + width; ++x)
    {
        uchar* row_ptr = img.ptr<uchar>(x - deb.x);

        for (int y = deb.y ; y < deb.y + height; ++y)
        {


            if(Matlabled.at<uint16_t>(Point(x,y)) == cc.getId_label()) row_ptr[y - deb.y] = 255;
            else row_ptr[y - deb.y] = 0;

            

        }
    }



    img = img.t();

   

}

void connectedComponentsVector(Mat& threshold_frame, vector<CC>& composants) {



    Mat centroids;
    

    Mat stats;
   connectedComponentsWithStats(threshold_frame, Matlabled, stats, centroids, 8, CV_16U);

    
    Mat m;
    for (int i = 0; i < stats.rows; i++)
    {
        CC composant;
        composant.setId_label(i);
        composant.setdX(stats.at<int>(i, 2));
        composant.setdY(stats.at<int>(i, 3));
        Point centroid(centroids.at<double>(i, 0), centroids.at<double>(i, 1));
        composant.setCentroid(centroid);
        composant.setPtr_debut(Point(stats.at<int>(i, 0), stats.at<int>(i, 1)));

        CcToMat(composant, m);
        
        composant.setMat(m);
        
        composants.push_back(composant);


    }
    
   



}



void capture(Mat& capture_frame,string path) {

    capture_frame = imread(path);

}



void filter(Mat& capture_frame, Mat& threshold_frame) {

    // capture frame, convert to grayscale, apply Gaussian blur, apply balance (if applicable), and apply adaptive threshold method
    cvtColor(capture_frame, filter_frame, COLOR_BGR2GRAY);
   GaussianBlur(filter_frame, gaussian_frame, cv::Size(kernel_size, kernel_size), segma, segma);
   // if (balance_flag) absdiff(gaussian_frame, balance_frame, gaussian_frame);
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