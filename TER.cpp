#include<opencv2/opencv.hpp>
#include<iostream>
#include<string>
#include <fstream>
#include <filesystem>
#include"math.h"
#include "CC.h"
#include "FRHistogram_Desc.c"
#include <iostream>
#include <stdio.h>
#include <io.h>

#define _CRT_SECURE_NO_DEPRECATE
#define M_PI 3.14159265358979323846
#define  MAX_DOUBLE 99999999999999999
#define  MAX_WIDTH_PGM 1000
#define  MAX_HEIGHT_PGM 1000

using namespace std;
using namespace cv;
namespace fs = std::filesystem;
using namespace cv::ml;

Mat train_data;
Mat train_labels;
Mat test_data;

Mat capture_frame, filter_frame, gaussian_frame, threshold_frame, centredImage;

vector<vector<CC>> matriceCompClassifier;

// vecteur des symoboles
vector<String> items;
// vecteur des distnaces
vector<double> distances;
//matrice avec des labels des composants connexes
Mat Matlabled;

// vecteur des composants connexes
vector<CC> composants;

// vectors des gfd (vector's des characteristic)
vector< vector<vector<float>>> vecteursCarPrim;
vector<vector<float>> vecteursCar;

// matricec dist
vector<double**> Mat_distances;

// variable tampon
vector<float> vectC;

bool balance_flag = false;
int kernel_size = 3;
int block_size = 3;
int c = 0;
double segma = 0;
double Seuil = 100;
int Seuil_distance = 100;
int connexité = 8;

int numberOfDirections = 180;

double rdep = 0.0;
double rfin = 0.0;
double rpas = 0.1;
double Seuil_similarity_ratio = 0.5;

//
int M = 4;
int N = 9;
// en moin 1 composant clasifier comme symbole qlq
int C = 1;
int NmbrSymbole = 4;
int numImagesMaxParSymbole = 3;
//
string path_image = "testRapide1.tif";

void capture(Mat& capture_frame, string path);
inline bool exists(const std::string& name);
void funcGenerale();
void readOrLoad(int m, int n, String Extension);
void symbolTocomposantGfd(Mat& mat, int m, int n);
void filterNonInv(Mat& capture_frame, Mat& threshold_frame);
void filter(Mat& capture_frame, Mat& threshold_frame);
double distance(Point& p1, Point& p2);
double ManhattanDistance(vector<float>& a, vector<float>& b);
void circshift(Mat& out, const Point& delta);
Point GetCentroid(Mat& img);
void centreObject(Mat& img, Mat& centredImage);
void CcToMat(CC cc, Mat& img);
void connectedComponentsVector(Mat& threshold_frame, vector<CC>& composants);
double GetExtrema(Mat& img, Point& center);
vector<double> linspace(double min, double max, int n);
void generate_Mat_distances();

void drawComposant_ligne(CC& composant, Mat& sub, int r, int g, int b);

int* near_4Composantes(int indexSymbole, int indexCCAccu);

static void meshgrid(const cv::Mat& xgv, const cv::Mat& ygv, cv::Mat1d& X, cv::Mat1d& Y);
void GFD(CC& composant, Mat& centredImage, int m, int n);
void CalculateGfdAndPushAllVectsCar(int m, int n);
void classification();
void clean_SomeShit();
void CC2PGM(CC& composant, String path);
vector<double>::iterator closest(std::vector<double>& vec, double value);

void CompClassifier2PGM(string path, string symbolename, vector<CC>& composantsDejaclassifier);

void drawComposant(CC& composant, Mat& sub);
void fill_Vector_mat_distances();
void drawComposantsClassifier(vector<CC>& composantsDejaclassifier, Mat& sub);
double* calcul_Histogram(string path_res, string path_image1, string path_image2);
double similarity_ratio(double* histo1, double* histo2);
double distance2CC(CC& CC1, CC& CC2);
void funcGenerale();

int main()
{
	clean_SomeShit();

	readOrLoad(M, N, ".png");

	capture(capture_frame, path_image);
	filter(capture_frame, threshold_frame);

	connectedComponentsVector(threshold_frame, composants);

	CalculateGfdAndPushAllVectsCar(4, 9);

	classification();

	funcGenerale();

	/*for (int index = 0; index < Mat_distances.size(); index++)
	{
		for (int i = 0; i < matriceCompClassifier.at(index).size(); i++)
		{
			cout << endl;
			for (int j = 0; j < matriceCompClassifier.at(index).size(); j++)
			{
				cout << Mat_distances[index][i][j] << " ";
			}
		}
	}*/

	waitKey(0);

	return 0;
}

double* calcul_Histogram(string path_res, string path_image1, string path_image2) {
	char res[100];
	char image1[100];
	char image2[100];

	strcpy(res, path_res.c_str());
	strcpy(image1, path_image1.c_str());
	strcpy(image2, path_image2.c_str());
	double* histo = FRHistogram(res, numberOfDirections, rdep, rfin, rpas, image1, image2);

	return histo;
}

double similarity_ratio(double* histo1, double* histo2) {
	vector<double> histo_1(histo1, histo1 + numberOfDirections);
	vector<double> histo_2(histo2, histo2 + numberOfDirections);
	double sumMin = 0.0;
	double sumMax = 0.0;

	double max_norm_1 = 0;
	double max_norm_2 = 0;
	for (int i = 0; i < histo_1.size() - 1; i++)
	{
		max_norm_1 += max(histo_2[i], histo_2[i + 1]);
		max_norm_2 += max(histo_1[i], histo_1[i + 1]);
	}

	double max;
	max = std::max(max_norm_1, max_norm_2);

	for (int i = 0; i < histo_1.size(); i++)
	{
		sumMin += min(histo_1[i] / max, histo_2[i] / max);
		sumMax += std::max(histo_1[i] / max, histo_2[i] / max);
	}

	return sumMin / sumMax;
}

void capture(Mat& capture_frame, string path) {
	capture_frame = imread(path);
}

inline bool exists(const std::string& name) {
	struct stat buffer;
	return (stat(name.c_str(), &buffer) == 0);
}

void filterNonInv(Mat& capture_frame, Mat& threshold_frame) {
	// capture frame, convert to grayscale, apply Gaussian blur, apply balance (if applicable), and apply adaptive threshold method
	cvtColor(capture_frame, filter_frame, COLOR_BGR2GRAY);
	GaussianBlur(filter_frame, gaussian_frame, cv::Size(kernel_size, kernel_size), segma, segma);
	threshold(gaussian_frame, threshold_frame, Seuil, 255, cv::THRESH_BINARY);
	//imwrite("./bin.jpg", threshold_frame);
}

void filter(Mat& capture_frame, Mat& threshold_frame) {
	// capture frame, convert to grayscale, apply Gaussian blur, apply balance (if applicable), and apply adaptive threshold method
	cvtColor(capture_frame, filter_frame, COLOR_BGR2GRAY);
	GaussianBlur(filter_frame, gaussian_frame, cv::Size(kernel_size, kernel_size), segma, segma);
	threshold(gaussian_frame, threshold_frame, Seuil, 255, cv::THRESH_BINARY_INV);
	//imwrite("./bin.jpg", threshold_frame);
}

double distance(Point& p1, Point& p2)
{
	double res;
	res = sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p1.y));
	return res;
}

double ManhattanDistance(vector<float>& a, vector<float>& b) {
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
		if (tmp0.size() != Size(0, 0))
			tmp0.copyTo(planes[i](Rect(0, y, sz.width, sz.height - y)));
		if (tmp1.size() != Size(0, 0))
			tmp1.copyTo(planes[i](Rect(0, 0, sz.width, y)));

		Mat q2(planes[i], Rect(0, 0, sz.width - x, sz.height));
		Mat q3(planes[i], Rect(sz.width - x, 0, x, sz.height));
		q2.copyTo(tmp2);
		q3.copyTo(tmp3);
		if (tmp2.size() != Size(0, 0))
			tmp2.copyTo(planes[i](Rect(x, 0, sz.width - x, sz.height)));
		if (tmp3.size() != Size(0, 0))
			tmp3.copyTo(planes[i](Rect(0, 0, x, sz.height)));
	}

	merge(planes, out);
}

Point GetCentroid(Mat& img) {
	Moments m = moments(img, true);

	return  Point((int)m.m10 / m.m00, (int)m.m01 / m.m00);
}

void centreObject(Mat& img, Mat& centredImage) {
	int width = img.size().width;
	int height = img.size().height;

	auto type = img.type();

	Mat ligne = cv::Mat::zeros(1, width, type);
	Mat colonne = cv::Mat::zeros(height, 1, type);

	centredImage = img;

	int temp = (int)(std::max(width, height) - std::min(width, height)) * 0.5;

	if (height < width) {
		if ((temp % 1) > 0) {
			hconcat(centredImage, colonne, centredImage);
			ligne = cv::Mat::zeros(1, centredImage.size().width, type);
		}

		for (int i = 0; i < round(temp); i++)
		{
			vconcat(centredImage, ligne, centredImage);
		}
	}
	else if (width < height) {
		if ((temp % 1) > 0) {
			vconcat(centredImage, ligne, centredImage);
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
	width = centredImage.size().width;
	ligne = cv::Mat::zeros(1, width, type);

	for (int i = 0; i < delta_max + 10; i++) {
		vconcat(ligne, centredImage, centredImage);
		vconcat(centredImage, ligne, centredImage);
	}

	circshift(centredImage, Point(delta_y, delta_x));
}

void CcToMat(CC cc, Mat& img) {
	int width = cc.getdX();
	int height = cc.getdY();

	img = cv::Mat(width, height, CV_8UC1);

	int longeur = Matlabled.size().width;

	Point deb = cc.getPtr_debut();

	for (int x = deb.x; x < deb.x + width; ++x)
	{
		uchar* row_ptr = img.ptr<uchar>(x - deb.x);

		for (int y = deb.y; y < deb.y + height; ++y)
		{
			if (Matlabled.at<uint16_t>(Point(x, y)) == cc.getId_label()) row_ptr[y - deb.y] = 255;
			else row_ptr[y - deb.y] = 0;
		}
	}

	img = img.t();
}

void connectedComponentsVector(Mat& threshold_frame, vector<CC>& composants) {
	Mat centroids;

	Mat stats;
	connectedComponentsWithStats(threshold_frame, Matlabled, stats, centroids, connexité, CV_16U);

	Mat m;
	composants.clear();
	for (int i = 0; i < stats.rows; i++)
	{
		CC composant;
		composant.setId_label(i);
		composant.setdX(stats.at<int>(i, 2));
		composant.setdY(stats.at<int>(i, 3));
		Point centroid((int)centroids.at<double>(i, 0), (int)centroids.at<double>(i, 1));
		composant.setCentroid(centroid);
		composant.setPtr_debut(Point(stats.at<int>(i, 0), stats.at<int>(i, 1)));

		CcToMat(composant, m);

		composant.setMat(m);

		composants.push_back(composant);

		imwrite("./CCs/composant_" + to_string(i) + ".jpg", composant.getMat());
	}
}

//still background
double GetExtrema(Mat& img, Point& center) {
	// Find contours
	vector<vector<Point>> cnts;

	cv::findContours(img, cnts, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

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

vector<double> linspace(double min, double max, int n)
{
	vector<double> result;
	// vector iterator
	int iterator = 0;

	for (int i = 0; i <= n - 2; i++)
	{
		double temp = min + i * (max - min) / (floor((double)n) - 1);
		result.insert(result.begin() + iterator, temp);
		iterator += 1;
	}

	//iterator += 1;

	result.insert(result.begin() + iterator, max);
	return result;
}

static void meshgrid(const cv::Mat& xgv, const cv::Mat& ygv,
	cv::Mat1d& X, cv::Mat1d& Y)
{
	cv::repeat(xgv.reshape(1, 1), ygv.total(), 1, X);
	cv::repeat(ygv.reshape(1, 1).t(), 1, xgv.total(), Y);
}

void GFD(CC& composant, Mat& centredImage, int m, int n)
{
	centreObject(composant.getMat(), centredImage);

	cout << "width : " << centredImage.size().width << endl;
	cout << "height : " << centredImage.size().height << endl;

	Point Centroid = GetCentroid(centredImage);

	cout << Centroid << endl;

	double maxRad = GetExtrema(centredImage, Centroid);

	double radius, tempR, tempI;
	double theta;

	int N = centredImage.size().height;
	/*vector<double> x = linspace(-N / 2, N / 2, N);
	Mat1d X;
	Mat1d Y;
	meshgrid(cv::Mat(x), cv::Mat(x), X, Y);

	for (size_t i = 0; i < X.size().width; i++)
	{
		for (size_t i = 0; i < X.size().height; i++)
		{
		}
	}*/

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
					radius = sqrt(std::pow(x - Centroid.x, 2) + std::pow(y - Centroid.y, 2)) / maxRad;
					theta = atan2((y - Centroid.y), (x - Centroid.x));
					if (theta < 0) theta += 2 * M_PI;

					tempR = centredImage.at<uchar>(Point(x, y)) * std::cos(2 * M_PI * rad * (radius)+ang * theta);
					tempI = centredImage.at<uchar>(Point(x, y)) * std::sin(2 * M_PI * rad * (radius)+ang * theta);

					FR.at(rad).at(ang) += tempR;
					FI.at(rad).at(ang) -= tempI;
				}
			}
		}
	}

	int taille = (m) * (n);
	vectC.clear();
	vectC.resize(taille);
	float DC;

	for (int rad = 0; rad < m; rad++)
	{
		for (int ang = 0; ang < n; ang++)
		{
			if (rad == 0 && ang == 0) {
				DC = sqrt(std::pow(FR.at(0).at(0), 2) + std::pow(FR.at(0).at(0), 2));

				vectC.at(0) = DC / (M_PI * std::pow(maxRad, 2));
			}
			else {
				int pos = rad * n + ang;
				vectC.at(pos) = sqrt(std::pow(FR.at(rad).at(ang), 2) + std::pow(FI.at(rad).at(ang), 2)) / DC;
			}
		}
	}
}

void CalculateGfdAndPushAllVectsCar(int m, int n) {
	vector<float>& ptr_vect = vectC;

	// i=0 => background
	for (int i = 1; i < composants.size(); i++)
	{
		cout << "calcul GFD Composant_" + to_string(i) << endl;
		GFD(composants.at(i), centredImage, m, n);

		//namedWindow("composantCentred : " + i, WINDOW_NORMAL);
		//imshow("composantCentred : " + i, centredImage);

		vecteursCar.push_back(ptr_vect);
	}
}

void clean_SomeShit() {
	std::error_code errorCode;
	std::filesystem::path dir = fs::current_path();

	std::filesystem::remove_all(dir / "CCs/", errorCode);
	std::filesystem::remove_all(dir / "PGM Files/", errorCode);
}

void drawComposant(CC& composant, Mat& sub) {
	//Mat sub = cv::Mat::zeros(100, 82, CV_8UC1);

	for (int x = composant.getPtr_debut().x; x < composant.getPtr_debut().x + composant.getdX(); ++x)
	{
		uchar* row_ptr = sub.ptr<uchar>(x);
		for (int y = composant.getPtr_debut().y; y < composant.getPtr_debut().y + composant.getdY(); ++y)
		{
			if (composant.getMat().at<uchar>(Point(x - composant.getPtr_debut().x, y - composant.getPtr_debut().y)) == 255) row_ptr[y] = 255;
		}
	}
}

void drawComposantsClassifier(vector<CC>& composantsDejaclassifier, Mat& sub) {
	for (int i = 0; i < composantsDejaclassifier.size(); i++)
	{
		drawComposant(composantsDejaclassifier.at(i), sub);
	}

	sub = sub.t();
}

void classification() {
	std::ofstream out("./resultat.txt");

	double min = MAX_DOUBLE;
	int symbole;
	double max = -MAX_DOUBLE;
	double dist = MAX_DOUBLE;

	matriceCompClassifier.clear();
	int N = vecteursCarPrim.size();
	matriceCompClassifier.resize(N, std::vector<CC>(C));

	for (int i = 0; i < vecteursCar.size(); i++)
	{
		cout << "composant_" << i + 1 << " =>";
		out << "composant_" << i + 1 << " =>";
		distances.clear();
		symbole = 0;

		for (int it = 0; it < vecteursCarPrim.at(0).size(); it++)
		{
			dist = std::min(dist, ManhattanDistance(vecteursCar.at(i), vecteursCarPrim.at(0).at(it)));
		}

		cout << items.at(0) << " " << dist << " | ";
		out << items.at(0) << " " << dist << " | ";
		distances.emplace_back(dist);
		max = dist;
		min = dist;

		for (int j = 1; j < vecteursCarPrim.size(); j++) {
			dist = MAX_DOUBLE;

			for (int it = 0; it < vecteursCarPrim.at(j).size(); it++)
			{
				dist = std::min(dist, ManhattanDistance(vecteursCar.at(i), vecteursCarPrim.at(j).at(it)));
			}

			if (dist <= min) {
				min = dist;
				symbole = j;
			}
			if (max <= dist) {
				max = dist;
			}

			cout << items.at(j) << " " << dist << " | ";
			out << items.at(j) << " " << dist << " | ";
			distances.emplace_back(dist);
		}
		std::sort(distances.begin(), distances.end());
		vector<double>::iterator itelt = closest(distances, min + ((100 - Seuil_distance) * max / 100));
		cout << " => " << *itelt;
		out << " => " << *itelt;

		cout << " => " << items.at((Seuil_distance != 100) ? (itelt - distances.begin()) : symbole);
		out << " => " << items.at((Seuil_distance != 100) ? (itelt - distances.begin()) : symbole);

		cout << endl;
		out << endl;

		matriceCompClassifier.at(symbole).emplace_back(composants.at(i + 1));
	}

	Mat image = cv::Mat::zeros(capture_frame.size().width, capture_frame.size().height, CV_8UC1);
	for (int i = 0; i < matriceCompClassifier.size(); i++)
	{
		string path = "PGM Files/" + items.at(i);
		string path2 = path + "/Lines/";
		fs::create_directories(path);
		fs::create_directories(path2);
		CompClassifier2PGM(path, items.at(i), matriceCompClassifier.at(i));
		image = cv::Mat::zeros(capture_frame.size().width, capture_frame.size().height, CV_8UC1);
		drawComposantsClassifier(matriceCompClassifier.at(i), image);
		imwrite("./CCs Classifier/" + items.at(i) + ".jpg", image);
	}
}

int* near_4Composantes(int indexSymbole, int indexCCAccu) {
	double temp = MAX_DOUBLE;
	int pos1 = -1, pos2 = -1, pos3 = -1, pos4 = -1;
	double firstmin = MAX_DOUBLE, secmin = MAX_DOUBLE, thirdmin = MAX_DOUBLE, fourthmin = MAX_DOUBLE;

	for (int j = 0; j < matriceCompClassifier.at(indexSymbole).size(); j++)
	{
		if (temp > Mat_distances[indexSymbole][indexCCAccu][j] && Mat_distances[indexSymbole][indexCCAccu][j] != -1) {
			temp = Mat_distances[indexSymbole][indexCCAccu][j];
			pos1 = j;
		}

		if (Mat_distances[indexSymbole][indexCCAccu][j] < firstmin && Mat_distances[indexSymbole][indexCCAccu][j] != -1)
		{
			fourthmin = thirdmin;
			thirdmin = secmin;
			secmin = firstmin;
			firstmin = Mat_distances[indexSymbole][indexCCAccu][j];
			pos1 = j;
		}

		else if (Mat_distances[indexSymbole][indexCCAccu][j] < secmin && Mat_distances[indexSymbole][indexCCAccu][j] != -1)
		{
			fourthmin = thirdmin;
			thirdmin = secmin;
			secmin = Mat_distances[indexSymbole][indexCCAccu][j];
			pos2 = j;
		}

		else if (Mat_distances[indexSymbole][indexCCAccu][j] < thirdmin && Mat_distances[indexSymbole][indexCCAccu][j] != -1) {
			fourthmin = thirdmin;
			thirdmin = Mat_distances[indexSymbole][indexCCAccu][j];
			pos3 = j;
		}
		else if (Mat_distances[indexSymbole][indexCCAccu][j] < fourthmin && Mat_distances[indexSymbole][indexCCAccu][j] != -1) {
			fourthmin = Mat_distances[indexSymbole][indexCCAccu][j];
			pos4 = j;
		}
	}

	int* c = new int[4];
	c[0] = pos1;
	c[1] = pos2;
	c[2] = pos3;
	c[3] = pos4;

	return c;
}

void generate_Mat_distances() {
	for (int i = 0; i < matriceCompClassifier.size(); i++)
	{
		Mat_distances.push_back(new double* [matriceCompClassifier.at(i).size()]);
		for (int j = 0; j < matriceCompClassifier.at(i).size(); j++)
		{
			Mat_distances[i][j] = new double[matriceCompClassifier.at(i).size()];
		}
	}
	fill_Vector_mat_distances();
}

double distance2CC(CC& CC1, CC& CC2) {
	if (CC1.getMat().empty() || CC2.getMat().empty()) return -1.0;
	Point p_debut_CC1 = CC1.getPtr_debut();
	Point p_debut_CC2 = CC2.getPtr_debut();
	int dx_p_debut_CC1 = CC1.getdX();
	int dy_p_debut_CC1 = CC1.getdY();

	int dx_p_debut_CC2 = CC2.getdX();
	int dy_p_debut_CC2 = CC2.getdY();

	Point p_fin_cc2_h = Point(p_debut_CC2.x + dx_p_debut_CC2, p_debut_CC2.y);
	Point p_fin_cc2_v = Point(p_debut_CC2.x, p_debut_CC2.y + dy_p_debut_CC2);
	Point p_fin_cc2_hv = Point(p_debut_CC2.x + dx_p_debut_CC2, p_debut_CC2.y + dy_p_debut_CC2);

	Point p_fin_cc1_h = Point(p_debut_CC1.x + dx_p_debut_CC1, p_debut_CC1.y);
	Point p_fin_cc1_v = Point(p_debut_CC1.x, p_debut_CC1.y + dy_p_debut_CC1);
	Point p_fin_cc1_hv = Point(p_debut_CC1.x + dx_p_debut_CC1, p_debut_CC1.y + dy_p_debut_CC1);

	double tab[8];

	tab[0] = distance(p_debut_CC1, p_fin_cc2_h);
	tab[1] = distance(p_debut_CC1, p_fin_cc2_v);
	tab[2] = distance(p_fin_cc1_v, p_fin_cc2_hv);
	tab[3] = distance(p_fin_cc1_v, p_debut_CC2);

	tab[4] = distance(p_fin_cc1_h, p_debut_CC2);
	tab[5] = distance(p_fin_cc1_h, p_fin_cc2_hv);

	tab[6] = distance(p_fin_cc1_hv, p_fin_cc2_v);
	tab[7] = distance(p_fin_cc1_hv, p_fin_cc2_h);

	return *min_element(tab, tab + 8);
}

void fill_mat_distances(int index) {
	for (int i = 0; i < matriceCompClassifier.at(index).size(); i++)
	{
		for (int j = 0; j < matriceCompClassifier.at(index).size(); j++)
		{
			Mat_distances[index][i][j] = -1;
			if (i != j)
				Mat_distances[index][i][j] = distance2CC(matriceCompClassifier.at(index).at(i), matriceCompClassifier.at(index).at(j));
		}
	}
}
void fill_Vector_mat_distances() {
	for (int i = 0; i < Mat_distances.size(); i++)
	{
		fill_mat_distances(i);
	}
}

bool exist(vector<CC>& vect_composants, int indexCC, int symbole) {
	return std::find(vect_composants.begin(), vect_composants.end(), matriceCompClassifier.at(symbole).at(indexCC)) != vect_composants.end();
}
void classifier_ligne(int indexCC, int symbole) {
	vector<CC> vect_composants = matriceCompClassifier.at(symbole);

	if (exist(vect_composants, indexCC, symbole)) {
		int* arry = near_4Composantes(symbole, indexCC);
		int b, c;

		if (arry[0] != -1 && arry[1] != -1)
			if (exist(vect_composants, arry[0], symbole) && exist(vect_composants, arry[1], symbole)) { b = arry[0]; c = arry[1]; }
			else if (arry[2] != -1 && arry[3] != -1)
				if (exist(vect_composants, arry[2], symbole) && exist(vect_composants, arry[3], symbole)) { b = arry[2]; c = arry[3]; }

		string path_res_1 = "./PGM Files/" + items.at(symbole) + "_" + to_string(indexCC) + "&" + items.at(symbole) + "_" + to_string(b) + ".txt";
		string path_res_2 = "./PGM Files/" + items.at(symbole) + "_" + to_string(c) + "&" + items.at(symbole) + "_" + to_string(indexCC) + ".txt";
		string path_image_1 = "./PGM Files/" + items.at(symbole) + "/" + items.at(symbole) + "_" + to_string(indexCC) + ".pgm";
		string path_image_2 = "./PGM Files/" + items.at(symbole) + "/" + items.at(symbole) + "_" + to_string(b) + ".pgm";
		string path_image_3 = "./PGM Files/" + items.at(symbole) + "/" + items.at(symbole) + "_" + to_string(c) + ".pgm";

		double* histo_1 = calcul_Histogram(path_res_1, path_image_1, path_image_2);
		double* histo_2 = calcul_Histogram(path_res_2, path_image_3, path_image_1);
		double d = similarity_ratio(histo_1, histo_2);
		if (d >= Seuil_similarity_ratio) {
			vect_composants.erase(remove(vect_composants.begin(), vect_composants.end(), matriceCompClassifier.at(symbole).at(indexCC)), vect_composants.end());
			vect_composants.erase(remove(vect_composants.begin(), vect_composants.end(), matriceCompClassifier.at(symbole).at(b)), vect_composants.end());
			vect_composants.erase(remove(vect_composants.begin(), vect_composants.end(), matriceCompClassifier.at(symbole).at(c)), vect_composants.end());
			Mat image = cv::Mat::zeros(MAX_WIDTH_PGM, MAX_HEIGHT_PGM, CV_8UC3);

			// draw 3 composants
			drawComposant_ligne(matriceCompClassifier.at(symbole).at(indexCC), image, 0, 255, 0);
			drawComposant_ligne(matriceCompClassifier.at(symbole).at(b), image, 0, 0, 255);
			drawComposant_ligne(matriceCompClassifier.at(symbole).at(c), image, 0, 0, 255);

			image = image.t();

			//save .jpg
			string name_image = items.at(symbole) + "[" + to_string(indexCC) + "_" + to_string(b) + "_" + to_string(c) + "]";
			imwrite("./PGM Files/" + items.at(symbole) + "/Lines/" + name_image + ".jpg", image);

			//save .pgm
			vector<int> compression_params;
			compression_params.push_back(IMWRITE_PXM_BINARY);
			compression_params.push_back(0);
			cvtColor(image, image, COLOR_RGB2GRAY);
			string path = "./PGM Files/" + items.at(symbole) + "/Lines/";
			imwrite(path + name_image + ".pgm", image, compression_params);

			propagation(vect_composants, name_image, b, c, symbole);
		}
	}
}

void propagation(vector<CC>& vect_composants, string name_image_without_Ex, int b, int c, int symbole) {
	int* arry_1 = near_4Composantes(symbole, b);
	int* arry_2 = near_4Composantes(symbole, c);
	int b, c;
	string path = "./PGM Files/" + items.at(symbole) + "/Lines/";
	//priority to b
	if (Mat_distances[symbole][b][arry_1[0]] < Mat_distances[symbole][c][arry_2[0]]) {
		if (exist(vect_composants, arry_1[0], symbole) && exist(vect_composants, arry_1[1], symbole)) {
			b = arry_1[0]; c = arry_1[1];
		}
		else if (Mat_distances[symbole][b][arry_1[2]] < Mat_distances[symbole][c][arry_2[0]]) {
			if (exist(vect_composants, arry_1[2], symbole) && exist(vect_composants, arry_1[3], symbole)) {
				b = arry_1[2]; c = arry_1[3];
			}
			else if (exist(vect_composants, arry_2[0], symbole) && exist(vect_composants, arry_2[1], symbole))
			{
				b = arry_2[0]; c = arry_2[1];
			}
			else if (exist(vect_composants, arry_2[2], symbole) && exist(vect_composants, arry_2[3], symbole))
			{
				b = arry_2[2]; c = arry_2[3];
			}
			else return;
		}
	}
	//priority to c
	else {
		if (exist(vect_composants, arry_2[0], symbole) && exist(vect_composants, arry_2[1], symbole)) {
			b = arry_2[0]; c = arry_2[1];
		}
		else if (Mat_distances[symbole][b][arry_2[2]] < Mat_distances[symbole][c][arry_1[0]]) {
			if (exist(vect_composants, arry_2[2], symbole) && exist(vect_composants, arry_2[3], symbole)) {
				b = arry_2[2]; c = arry_2[3];
			}
			else if (exist(vect_composants, arry_1[0], symbole) && exist(vect_composants, arry_1[1], symbole))
			{
				b = arry_1[0]; c = arry_1[1];
			}
			else if (exist(vect_composants, arry_1[2], symbole) && exist(vect_composants, arry_1[3], symbole))
			{
				b = arry_1[2]; c = arry_1[3];
			}
			else return;
		}
	}

	if (b == -1 || c == -1) return;

	string path_res_1 = "./PGM Files/" + items.at(symbole) + "_" + name_image_without_Ex + "&" + items.at(symbole) + "_" + to_string(b) + ".txt";
	string path_res_2 = "./PGM Files/" + items.at(symbole) + "_" + to_string(c) + "&" + items.at(symbole) + "_" + name_image_without_Ex + ".txt";
	string path_image_1 = "./PGM Files/" + items.at(symbole) + "/" + items.at(symbole) + "_" + name_image_without_Ex + ".pgm";
	string path_image_2 = "./PGM Files/" + items.at(symbole) + "/" + items.at(symbole) + "_" + to_string(b) + ".pgm";
	string path_image_3 = "./PGM Files/" + items.at(symbole) + "/" + items.at(symbole) + "_" + to_string(c) + ".pgm";

	double* histo_1 = calcul_Histogram(path_res_1, path_image_1, path_image_2);
	double* histo_2 = calcul_Histogram(path_res_2, path_image_3, path_image_1);
	double d = similarity_ratio(histo_1, histo_2);

	if (d >= Seuil_similarity_ratio) {
		//vect_composants.erase(remove(vect_composants.begin(), vect_composants.end(), matriceCompClassifier.at(symbole).at(indexCC)), vect_composants.end());
		vect_composants.erase(remove(vect_composants.begin(), vect_composants.end(), matriceCompClassifier.at(symbole).at(b)), vect_composants.end());
		vect_composants.erase(remove(vect_composants.begin(), vect_composants.end(), matriceCompClassifier.at(symbole).at(c)), vect_composants.end());
		Mat image = cv::Mat::zeros(MAX_WIDTH_PGM, MAX_HEIGHT_PGM, CV_8UC3);

		// draw 3 composants
		drawComposant_ligne(matriceCompClassifier.at(symbole).at(indexCC), image, 0, 255, 0);
		drawComposant_ligne(matriceCompClassifier.at(symbole).at(b), image, 0, 0, 255);
		drawComposant_ligne(matriceCompClassifier.at(symbole).at(c), image, 0, 0, 255);

		image = image.t();

		//save .jpg
		string name_image = items.at(symbole) + "[" + to_string(indexCC) + "_" + to_string(b) + "_" + to_string(c) + "]";
		imwrite("./PGM Files/" + items.at(symbole) + "/Lines/" + name_image + ".jpg", image);

		//save .pgm
		vector<int> compression_params;
		compression_params.push_back(IMWRITE_PXM_BINARY);
		compression_params.push_back(0);
		cvtColor(image, image, COLOR_RGB2GRAY);
		string path = "./PGM Files/" + items.at(symbole) + "/Lines/";
		imwrite(path + name_image + ".pgm", image, compression_params);

		propagation(vect_composants, name_image, b, c, symbole);
	}
}
void drawComposant_ligne(CC& composant, Mat& sub, int r, int g, int b) {
	for (int x = composant.getPtr_debut().x; x < composant.getPtr_debut().x + composant.getdX(); ++x)
	{
		cv::Vec3b* row_ptr = sub.ptr<cv::Vec3b>(x);
		for (int y = composant.getPtr_debut().y; y < composant.getPtr_debut().y + composant.getdY(); ++y)
		{
			if (composant.getMat().at<uchar>(Point(x - composant.getPtr_debut().x, y - composant.getPtr_debut().y)) == 255) row_ptr[y] = cv::Vec3b(uchar(r), uchar(g), uchar(b));
		}
	}
}
void funcGenerale() {
	generate_Mat_distances();

	for (int i = 0; i < matriceCompClassifier.size(); i++)
	{
		if (matriceCompClassifier.at(i).size() >= 4)classifier_ligne(1, i);
	}
}

void CompClassifier2PGM(string path, string symbolename, vector<CC>& composantsDejaclassifier) {
	for (int i = 0; i < composantsDejaclassifier.size(); i++)
	{
		CC2PGM(composantsDejaclassifier.at(i), path + "/" + symbolename + "_" + to_string(i) + ".pgm");
	}
}
void CC2PGM(CC& composant, String path) {
	vector<int> compression_params;
	compression_params.push_back(IMWRITE_PXM_BINARY);
	compression_params.push_back(0);
	Mat dst;
	if (!composant.getMat().empty()) {
		Size size(MAX_WIDTH_PGM, MAX_HEIGHT_PGM);
		resize(composant.getMat(), dst, size);

		imwrite(path, dst, compression_params);
	}
}

vector<double>::iterator closest(std::vector<double>& vec, double value) {
	vector<double>::iterator ite = std::find(vec.begin(), vec.end(), value);
	if (ite != vec.end()) return ite;
	auto const it = std::lower_bound(vec.begin(), vec.end(), value);
	if (it == vec.end()) {
		auto const it = std::upper_bound(vec.begin(), vec.end(), value);
		if (it != vec.end()) return it;
		return it;
	}

	return it;
}

void readOrLoad(int m, int n, String Extension) {
	items.clear();
	String numS;
	String path;
	vecteursCarPrim.clear();
	vecteursCarPrim.resize(NmbrSymbole, vector<std::vector<float>>(C));
	for (auto& p : fs::directory_iterator("Symboles")) {
		items.push_back(p.path().filename().string());
		for (int num = 0; num < numImagesMaxParSymbole; num++)
		{
			numS = "_" + to_string(num);
			path = "Symboles/" + p.path().filename().string() + "\/" + p.path().filename().string() + numS;

			if (!exists(path + Extension)) continue;
			if (exists(path + ".txt")) {
				std::ifstream file(path + ".txt");

				std::string str;

				std::getline(file, str);
				std::istringstream iss(str);
				int a, b;

				if (!(iss >> a >> b)) { break; }

				if (a == m & b == n) {
					std::getline(file, str);

					int pos = 0;
					std::string token;
					string delimiter = " ";
					vector<float> vect;

					while ((pos = str.find(delimiter)) != std::string::npos) {
						token = str.substr(0, pos);
						vect.push_back(stod(token));
						str.erase(0, pos + delimiter.length());
					}
					vecteursCarPrim.at(items.size() - 1).emplace_back(vect);
				}
				else
				{
					std::ofstream outfile(path + ".txt");
					int a = 0;
					int b = 0;
					outfile << m << " " << n << std::endl;

					Mat symbole;
					capture(symbole, path + Extension);

					cout << "calcul GFD du Symbole " + items.at(items.size() - 1) << endl;
					symbolTocomposantGfd(symbole, m, n);

					String str;
					vector<float>& ptr_vect = vectC;
					for (int i = 0; i < vectC.size(); i++)
					{
						str += to_string(ptr_vect.at(i)) + " ";
					}
					outfile << str;

					outfile.close();
					vecteursCarPrim.at(items.size() - 1).emplace_back(ptr_vect);
				}
			}
			else
			{
				std::ofstream outfile(path + ".txt");
				int a = 0;
				int b = 0;
				outfile << m << " " << n << std::endl;

				Mat symbole;
				capture(symbole, path + Extension);

				cout << "calcul GFD du Symbole " + items.at(items.size() - 1) << endl;
				symbolTocomposantGfd(symbole, m, n);

				String str;
				vector<float>& ptr_vect = vectC;
				for (int i = 0; i < vectC.size(); i++)
				{
					str += to_string(ptr_vect.at(i)) + " ";
				}
				outfile << str;

				outfile.close();
				vecteursCarPrim.at(items.size() - 1).emplace_back(ptr_vect);
			}
		}
	}
}

void symbolTocomposantGfd(Mat& mat, int m, int n) {
	Mat symbolecentred;

	filterNonInv(mat, mat);

	CC composant;
	composant.setId_label(-1);
	composant.setdX(mat.size().width);
	composant.setdY(mat.size().height);
	Point centroid((int)mat.size().width / 2, (int)mat.size().height / 2);
	composant.setCentroid(centroid);
	composant.setPtr_debut(Point(0, 0));
	composant.setMat(mat);

	GFD(composant, symbolecentred, m, n); //vectC
}