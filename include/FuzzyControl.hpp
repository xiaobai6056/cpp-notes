#ifndef CODE_FUZZY_H_
#define CODE_FUZZY_H_

#include <iostream>

//定义偏差论域
#define   Pos_NB   0   //负大
#define   Pos_NM   1   //负中
#define   Pos_NS   2   //负小
#define   Pos_ZO   3   //零
#define   Pos_PS   4   //正小
#define   Pos_PM   5   //正中
#define   Pos_PB   6   //正大
//定义偏差变化率论域
#define   Dpos_NB   0  //负大
#define   Dpos_NM   1  //负中
#define   Dpos_NS   2  //负小
#define   Dpos_ZO   3  //零
#define   Dpos_PS   4  //正小
#define   Dpos_PM   5  //正中
#define   Dpos_PB   6  //正大

/////////////////////////////////////
#define   Pos_NB_C   0.0f
#define   Pos_NB_R   1.0f

#define   Pos_NM_L   0.0f
#define   Pos_NM_C   1.0f
#define   Pos_NM_R   2.0f

#define   Pos_NS_L   1.0f
#define   Pos_NS_C   2.0f
#define   Pos_NS_R   3.0f

#define   Pos_ZO_L   2.0f
#define   Pos_ZO_C   3.0f
#define   Pos_ZO_R   4.0f

#define   Pos_PS_L   3.0f
#define   Pos_PS_C   4.0f
#define   Pos_PS_R   5.0f

#define   Pos_PM_L   4.0f
#define   Pos_PM_C   5.0f
#define   Pos_PM_R   6.0f

#define   Pos_PB_L   5.0f
#define   Pos_PB_C   6.0f

//////////////////////////////////////////
#define   Dpos_NB_C   0.0f//1.5f
#define   Dpos_NB_R   1.0f//2.0f

#define   Dpos_NM_L   0.0f//0.9f//1.5f
#define   Dpos_NM_C   1.0f//1.6f//2.0f
#define   Dpos_NM_R   2.0f//2.3f//2.5f

#define   Dpos_NS_L   1.0f//1.6f//
#define   Dpos_NS_C   2.0f//2.3f//
#define   Dpos_NS_R   3.0f//3.0f//

#define   Dpos_ZO_L   2.0f//2.3f//
#define   Dpos_ZO_C   3.0f//3.0f//
#define   Dpos_ZO_R   4.0f//3.7f//

#define   Dpos_PS_L   3.0f//3.0f//
#define   Dpos_PS_C   4.0f//3.7f//
#define   Dpos_PS_R   5.0f//4.4f//

#define   Dpos_PM_L   4.0f//3.7f//
#define   Dpos_PM_C   5.0f//4.4f//
#define   Dpos_PM_R   6.0f//5.1f//

#define   Dpos_PB_L   5.0f//4.4f//
#define   Dpos_PB_C   6.0f//5.1f//

// /////////////////P参数
// #define NL1 5.8
// #define NB1 4.6
// #define NM1 3.5
// #define NS1 2.5
// #define ZO1 1.5
// #define PS1 2.5
// #define PM1 3.5
// #define PB1 4.6
// #define PL1 5.8
////////////////P参数
// #define NL1 14.0f //18
// #define NB1 12.0f
// #define NM1 8.0f
// #define NS1 6.0f
// #define ZO1 4.0f
// #define PS1 6.0f
// #define PM1 8.0f
// #define PB1 12.0f
// #define PL1 14.0f //18

////////////////P参数
#define NL_1 18.0f //18
#define NB1 16.0f
#define NM1 10.0f
#define NS1 6.0f
#define ZO1 4.0f
#define PS1 6.0f
#define PM1 10.0f
#define PB1 16.0f
#define PL1 18.0f //18

//////////////////I参数
#define PB2  3
#define PM2  2
#define PS2  1
#define ZO2  0   
#define NS2  1
#define NM2  2
#define NB2  3
//////////////////D参数
#define PB3  18.0f  //18
#define PM3  14.0f
#define PS3  7.0f
#define ZO3  4.0f
#define NS3  7.0f
#define NM3  14.0f
#define NB3  18.0f  //18

// //////////////////D参数
// #define PB3  10.0f  //18
// #define PM3  8.0f
// #define PS3  6.0f
// #define ZO3  4.0f
// #define NS3  6.0f
// #define NM3  8.0f
// #define NB3  10.0f  //18


typedef struct Fuzzy//标识符
{

    //速度相关模糊变量

        float Pos; //定义论域
        float Dpos;
        int   Pos_Rule_Value[2];//定义偏差模糊值
        float Pos_U_Value[2];//定义偏差隶属度
        int   Dpos_Rule_Value[2];//定义偏差变化率模糊值
        float Dpos_U_Value[2];//定义偏差变化率隶属度
        int   Servo_Rule[4];//模糊推理舵机输出量
        int   Speed_Rule[4];//模糊推理电机输出量
        float Servo_U[4];       //模糊推理舵机隶属度
        float Speed_U[4];       //模糊推理电机隶属度

        float P_Rule[4];//模糊推理P输出量
        float I_Rule[4];//模糊推理I输出量
        float D_Rule[4];//模糊推理D输出量
        float P_U[4];       //模糊推理P隶属度
        float I_U[4];       //模糊推理I隶属度
        float D_U[4];       //模糊推理D隶属度

        float Speed_out_u;
        float Speed_out;//电机最终输出值
        float Servo_out_u;
        float Servo_out; //舵机最终输出值

        float P_out_u;
        float P_out;//舵机P最终输出值
        float I_out_u;
        float I_out; //舵机I最终输出值
        float D_out_u;
        float D_out;//舵机D最终输出值

}Fuzzy;//变量类型

class FuzzyControl{
private:

    struct Fuzzy Fz;
    float temp_dif;
    float d_temp_dif;
    float last_servo_out;//定义舵机最终输出值  

    
    float Bias;               //偏差值
    float Integral_bias;      //偏差积分
    float Last_Bias;          //上一次的偏差

public:
    void Fuzzy_init(void); //模糊初始化
    void Fuzzy_Pos_input(float s_error);    //偏差量化函数
    void Fuzzy_Dpos_input(float s_derror);  //偏差变化率量化函数
    void Pos_fuzzy(void);     //计算隶属度
    void Dpos_fuzzy(void);    //计算隶属度
    float Fuzzy_Kp(float servo_Err,float servo_lastErr);
    float Fuzzy_Ki(float servo_Err,float servo_lastErr);
    float Fuzzy_Kd(float servo_Err,float servo_lastErr);

    void PID_steering_out(float s_error,float s_derror); //舵机PID算法
    void FuzzyPID(float servo_Err,float servo_lastErr);    //模糊PID控制
    float steering_KP;    //舵机P值
    float basic_KP;       //基础P值
    float steering_KI;    //舵机I值
    float steering_KD;    //舵机D值
    float basic_KD;       //基础P值

};

#endif