#ifndef _SERVO_H_
#define _SERVO_H_
#include <iostream>
#include <fstream>
#include "FuzzyControl.hpp"
#include <cmath>
using namespace std;

class servo{
private:
	
public:
	servo(){
		outfile.open("/root/workspace/Mars3/res/errorData/errorDate.txt",ios::trunc);
	};
    float servoControl(float err,float alpha=0);
	float servo_Err = 0;					//位置式误差
	float servo_lastErr = 0;				//记录上次的误差
	FuzzyControl fuzzyer;
	bool FuzzyEnable=false;					//模糊控制器使能
	bool PolyEnable=false;					//多项式控制器使能
	float P3base;							//三次项p值
    float Pbase;							//基础p值
    float Ibase=0;							//基础i值
    float Dbase;							//基础d值
	float D3base;							//三次项d值
	float PdP3base;							//三次项p值
    float PdPbase;							//基础p值
    float PdDbase;							//基础d值
	float PdD3base;							//三次项d值
	float weight=0;							//误差调节P权重
	ofstream outfile;						//将误差数据输出到txt然后用py画图
	
    
};
#endif
