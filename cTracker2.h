#include<vector>
#include<opencv2/opencv.hpp>
#include "cBlob.h"
#include "cDisjointSet.h"

using namespace std;
using namespace cv;
class cTracker2 {
private:
    double min_area, max_radius;
    node** labels;
    cDisjointSet ds;

    

    // storage of the current blobs and the blobs from the previous frame
    vector<cBlob> blobs;

    

protected:

public:
    
    cTracker2(double min_area, double max_radius);
    ~cTracker2();

    void extractBlobs(cv::Mat& mat);
   
    node** getlabels();
    int zonesCount();
   
    vector<cBlob>& getBlobs();
};