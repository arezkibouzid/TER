#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class CC
{
private:

	Point ptr_debut;
	Point centroid;
	int id_label;
	int dX;
	int dY;
	Mat mat;

protected:

public:

	Point getPtr_debut() { return ptr_debut; }
	Point getCentroid() { return centroid; }

	int getId_label() { return id_label; }
	int getdX() { return dX; }
	int getdY() { return dY; }

	Mat& getMat() { return mat; }

	void setPtr_debut(Point p) { ptr_debut = p; }
	void setCentroid(Point p) { centroid = p; }

	void setId_label(int id) { id_label = id; }
	void setdX(int x) { dX = x; }
	void setdY(int y) { dY = y; }
	bool operator ==(const CC& b) {
		return this->ptr_debut.x == b.ptr_debut.x && this->centroid.x == b.centroid.x
			&& this->ptr_debut.y == b.ptr_debut.y && this->centroid.y == b.centroid.y && this->id_label == b.id_label && this->dX == b.dX && this->dY == b.dY;
	}

	void setMat(Mat m) { mat = m; }
};
