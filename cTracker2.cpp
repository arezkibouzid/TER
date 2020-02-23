#include "ctracker2.h"

#include<opencv2/opencv.hpp>
#include<iostream>
#include<vector>

cTracker2::cTracker2(double min_area, double max_radius) :
    min_area(min_area), max_radius(max_radius),
    labels(NULL) {
}

cTracker2::~cTracker2() {
    if (labels) delete[] labels;
}

node** cTracker2::getlabels() {

    return labels;
}

int cTracker2::zonesCount() {

    return ds.SetCount();
}
void cTracker2::extractBlobs(cv::Mat& mat) {


    int width = mat.cols; int height = mat.rows;
    if (this->labels) delete[] labels;
    int taille = width * height;
    this->labels = new node * [taille];


    // reset our data structure for reuse

    ds.Reset();
    int index;

    // generate equivalence sets -- connected component labeling (4-connected)

    this->labels[0] = ds.MakeSet(0);
    for (int j = 1; j < width; j++)
        this->labels[j] = mat.data[j] != mat.data[j - 1] ? ds.MakeSet(0) : this->labels[j - 1];

    
    for (int j = width; j < height * width; j++) {

        if (mat.data[j] == mat.data[j - 1]) {
            this->labels[j] = this->labels[j - 1];


            if (mat.data[j - 1] == mat.data[j - width]) ds.Union(this->labels[j - 1], this->labels[j - width]);
            //if (mat.data[j - 1] == mat.data[j - width-1]) ds.Union(this->labels[j - 1], this->labels[j - width-1]);
            //if (mat.data[j - 1] == mat.data[j + width]) ds.Union(this->labels[j - 1], this->labels[j + width] = ds.MakeSet(0));
        }
        else if (mat.data[j] == mat.data[j - width]) {
            this->labels[j] = this->labels[j - width];
            //if (mat.data[j-width] == mat.data[j + width]) this->labels[j] = this->labels[j + width];
            //if (mat.data[j] == mat.data[j - width - 1]) ds.Union(this->labels[j], this->labels[j - width - 1]);
            //if (mat.data[j] == mat.data[j + width]) ds.Union(this->labels[j], this->labels[j + width] = ds.MakeSet(0));
        }
       /* else if (mat.data[j] == mat.data[j - width - 1]) {
            mat.data[j] == mat.data[j - width - 1];

            if (mat.data[j] == mat.data[j + width]) ds.Union(this->labels[j], this->labels[j + width]= ds.MakeSet(0));

        }
        else if (mat.data[j] == mat.data[j + width]) {
            this->labels[j] = this->labels[j + width];
        }*/
        
        else this->labels[j] = ds.MakeSet(0);
    }


   



    // the representative elements in our disjoint set data struct are associated with indices
    // we reduce those indices to 0,1,...,n and allocate our blobs
    cBlob temp;
    temp.event = BLOB_NULL;
    blobs.clear();
    for (int i = 0; i < ds.Reduce(); i++)
        blobs.push_back(temp);
    node* pos = 0;
    // populate our blob vector
    for (int j = 0; j < mat.rows; j++) {
        for (int i = 0; i < mat.cols; i++) {
            pos = this->labels[j * mat.size().width + i];
            index = ds.Find(pos)->i;
            if (this->blobs[index].event == BLOB_NULL) {
                this->blobs[index].min.x = this->blobs[index].max.x = i;
                this->blobs[index].min.y = this->blobs[index].max.y = j;
                this->blobs[index].event = BLOB_DOWN;

            }
            else {
                if (this->blobs[index].min.x > i) this->blobs[index].min.x = i;
                else if (this->blobs[index].max.x < i) this->blobs[index].max.x = i;
                this->blobs[index].max.y = j;
            }
        }
    }



    // apply blob filter
    for (int i = 0; i < blobs.size(); i++) {
        if ((this->blobs[i].max.x - this->blobs[i].min.x) * (this->blobs[i].max.y - this->blobs[i].min.y) < min_area) { blobs.erase(blobs.begin() + i); i--; }
    }

    // find blob centers
    for (int i = 0; i < blobs.size(); i++) {
        this->blobs[i].location.x = (int)(this->blobs[i].max.x + this->blobs[i].min.x) / 2.0;
        this->blobs[i].location.y = (int)(this->blobs[i].max.y + this->blobs[i].min.y) / 2.0;


    }



}

vector<cBlob>& cTracker2::getBlobs() {
    return blobs;
}