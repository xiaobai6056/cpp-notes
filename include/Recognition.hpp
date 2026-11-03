#pragma once
#include<iostream>
#include<opencv2/opencv.hpp>
// #include"PathSearch.hpp"
#include"PurePersuit.hpp"
#include"Serial.hpp"
//#include "detection.hpp"
// #include <cmath>

using namespace std;
using namespace cv;

struct condition{
        // bool BothLFound;
        bool lStraight;
        bool rStraight;
        // bool lLinear;
        // bool rLinear;
        bool lLFound;
        bool rLFound;
        bool lLFound_doublecheck;
        bool rLFound_doublecheck;
        int lLPointIndex;
        int rLPointIndex;
        bool BrideUp;
        bool BrideDown;
        bool lParkingFound;
        bool rParkingFound;
        int  lParkingPointIndex;
        int  rParkingPointIndex;
};

class Recognition{
private:
    
    //bool checkCorner(PathSearch& pathSearcher,vector<PredictResult> results,int idx,int side);
    int countFrames =0;
public:
    Point2f   lYPoint;
    Point2f   rYPoint;
    Point2f   lLPoint;
    Point2f   rLPoint;
    condition conditions;
    Mat  homoMatrix;
    condition findCorners(Mat& binary,PathSearch& pathSearcher,SearchMode mode);
    void drawImage(Mat& outputImg);
    int clip(int x, int low, int up) {
        return x > up ? up : x < low ? low : x;
    };
    float fclip(float x, float low, float up) {
        return x > up ? up : x < low ? low : x;
    };
};
