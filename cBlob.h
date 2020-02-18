#ifndef DEF_CBLOB
#define DEF_CBLOB

// Code...


#include<iostream>
#include<opencv2/opencv.hpp>
using namespace std;
using namespace cv;


enum { BLOB_NULL, BLOB_DOWN };



struct point {
    

    double x, y;

    std::string toString() {
        return "(" + std::to_string(x) + "," + std::to_string(y)+")";
    }
};


class cBlob {
private:

protected:

public:
    point location ; // current location and origin for defining a drag vector
    point min, max;     // to define our axis-aligned bounding box
    int event;      // event type: one of BLOB_NULL, BLOB_DOWN, BLOB_MOVE, BLOB_UP

    

    std::ostream& operator<<(std::ostream& Str) {
        
        Str << "[" + location.toString() + "]";
        return Str;
    }
   
};

#endif