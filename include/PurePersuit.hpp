#pragma once
#include<iostream>
#include<opencv2/opencv.hpp>
#include<math.h>
using namespace std;
using namespace cv;


class PurePersuit{
private:
    enum RoadType
    {
        BaseHandle = 0, // 基础赛道处理
        RingHandle,     // 环岛赛道处理
        CrossHandle,    // 十字道路处理

        BridgeHandle,   // 坡道(桥)处理
        ParkingHandle,
        StartHandle,
        BombHandle,
        // FreezoneHandle, // 泛行区处理
        // GarageHandle,   // 车库处理
        // GranaryHandle,  // 粮仓处理
        // DepotHandle,    // 修车厂处理
        // SlowzoneHandle, // 慢行区（动物出没）处理
        // FarmlandHandle, // 农田区域处理
    };
    
   
public:
    void    selectAchor(vector<Point2f> & centerLine);
    Point2f computeWorldCoordinate(Point2f& targetInImage);
    float   computeTheta();
    float   ComputeSteeringAngle();
    void    drawImage(Mat& outputImage);
    
    
    float metersPerPixel=0.005357;
    float imageCols=320;
    float refDistance=0.43;
    float refYonImage=204;   
    float wheelBase=0.2;     // the distance between front wheel and rear wheel
    float aheadDistance=1.0; 
    // float aheadDistanceSet=1.0;
    Point2f targetInImage;
    Point2f targetInWorld;
    float theta=0;
    float steeringAngle=0;

    bool fixed=false; 
    float fixedAngle=30;

};