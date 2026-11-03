/*
 * fuzzy.c
 *
 *  Created on: 2022年2月26日
 *      Author: Admin
 */

#include "../include/FuzzyControl.hpp"
#include <iostream>
using namespace std;

float abc(float a)
{
    if(a>0)
    {
        return a;
    }
    else
        return a*(-1);
}


const float fuzzyRuleKp[7][7] =
{

      {NL_1,NB1,NM1,NB1,NM1,NS1,ZO1},
	    {NB1,NM1,NS1,NM1,ZO1,ZO1,NS1},
	    {NM1,NS1,NS1,NS1,ZO1,ZO1,NS1},
	    {ZO1,ZO1,ZO1,ZO1,ZO1,ZO1,ZO1},
   	  {PS1,ZO1,ZO1,PS1,PS1,PS1,PM1},
	    {PS1,ZO1,ZO1,PM1,PS1,PM1,PB1},
      {ZO1,PS1,PM1,PB1,PM1,PB1,PL1},    

};//dKp模糊规则表

const float fuzzyRuleKi[7][7]=
{
       {NB2,NB2,NM2,NM2,NS2,ZO2,ZO2},
	     {NB2,NB2,NM2,NS2,NS2,ZO2,ZO2},
			 {NB2,NM2,NS2,NS2,ZO2,PS2,PS2},
	     {NM2,NM2,NS2,ZO2,PS2,PM2,PM2},
	     {NM2,NS2,ZO2,PS2,PS2,PM2,PB2},
	     {ZO2,ZO2,PS2,PS2,PM2,PB2,PB2},
	     {ZO2,ZO2,PS2,PM2,PM2,PB2,PB2},

        
};//dKi模糊规则表


const float fuzzyRuleKd[7][7]={

    {NB3,NM3,NS3,ZO3,PS3,PM3,PB3},
	  {NB3,NM3,NS3,ZO3,PS3,PM3,PB3},
		{NB3,NM3,NS3,ZO3,PS3,PM3,PB3},
	  {NB3,NM3,NS3,ZO3,PS3,PM3,PB3},
	  {NB3,NM3,NS3,ZO3,PS3,PM3,PB3},
	  {NB3,NM3,NS3,ZO3,PS3,PM3,PB3},
	  {NB3,NM3,NS3,ZO3,PS3,PM3,PB3},

};//dKd模糊规则表




float min(float x,float y)
{
  float z;
  if(x<y) z=x;
  else z=y;
  return (z);
}

/**********输出最大值*****************/
float max(float x,float y)
{
  float z;
  if(x>y) z=x;
  else z=y;
  return (z);
}
void FuzzyControl::Fuzzy_init(void)//模糊控制初始化函数
{
    Fz.Pos=0;//定义论域
    Fz.Dpos=0;

    Fz.Pos_Rule_Value[0]=Pos_ZO;//定义偏差模糊值
    Fz.Pos_Rule_Value[1]=Pos_ZO;

    Fz.Pos_U_Value[0]=1;//定义偏差隶属度
    Fz.Pos_U_Value[1]=0;

    Fz.Dpos_Rule_Value[0]=Dpos_ZO;//定义偏差变化率模糊值
    Fz.Dpos_Rule_Value[1]=Dpos_ZO;

    Fz.Dpos_U_Value[0]=1;//定义偏差变化率隶属度
    Fz.Dpos_U_Value[1]=0;

    Fz.Servo_out = 0;//舵机输出值
    Fz.Speed_out = 0;//电机输出值
}

void FuzzyControl::Fuzzy_Pos_input(float s_error)    //偏差量化函数
{

  temp_dif=s_error;
   if(temp_dif>=30)
  {
       temp_dif =30;
  }
   if(temp_dif<=-30)
   {
       temp_dif = -30;
   }
    Fz.Pos =  (float)temp_dif/10+3 ;   // Pos[0 6]
}
void FuzzyControl::Fuzzy_Dpos_input(float s_derror)  //变化值偏差量化
{

    //变化量偏差量化为[0,6]
   d_temp_dif =s_derror;
  if(d_temp_dif>60)
   {
      d_temp_dif = 60;//确定最大EC=6
   }
   if(d_temp_dif<-60)
   {
      d_temp_dif =-60;//确定最小EC=0
   }
   Fz.Dpos = d_temp_dif/20+3;
   

}
void FuzzyControl::Pos_fuzzy(void)
{
    int i=0,j=0;
    float f1,f2;
    for(j=0;j<2;j++)
    {
      Fz.Pos_Rule_Value[j]=0;
    }//先将之前运算的隶属度变为0

    if(Fz.Pos < Pos_NB_R)        //论域1    01之间 模糊值为1 隶属度为
    {
      Fz.Pos_Rule_Value[i] = Pos_NB;  //确定模糊值

      f2 = (Pos_NB_R - Fz.Pos);
      f1 = f2;

      Fz.Pos_U_Value[i] = min(f1,f2);  //计算隶属度

      i++;
    }

    if((Fz.Pos>=Pos_NM_L)&&(Fz.Pos<Pos_NM_R))//论域2
    {
      Fz.Pos_Rule_Value[i]=Pos_NM;

      f1 = (Fz.Pos - Pos_NM_L);
      f2 = (Pos_NM_R - Fz.Pos);

      Fz.Pos_U_Value[i]=min(f1,f2);

      i++;
    }

    if((Fz.Pos>=Pos_NS_L)&&(Fz.Pos<Pos_NS_R))//论域3
    {
      Fz.Pos_Rule_Value[i]=Pos_NS;

      f1 = (Fz.Pos - Pos_NS_L);
      f2 = (Pos_NS_R - Fz.Pos);

      Fz.Pos_U_Value[i]=min(f1,f2);

      i++;
    }

    if((Fz.Pos>=Pos_ZO_L)&&(Fz.Pos<Pos_ZO_R))//论域4
    {
      Fz.Pos_Rule_Value[i]=Pos_ZO ;

      f1 = (Fz.Pos - Pos_ZO_L);
      f2 = (Pos_ZO_R - Fz.Pos);

      Fz.Pos_U_Value[i]=min(f1,f2);

      i++;
    }


    if((Fz.Pos>=Pos_PS_L)&&(Fz.Pos<Pos_PS_R))//论域5
    {
      Fz.Pos_Rule_Value[i]=Pos_PS;

      f1 = (Fz.Pos - Pos_PS_L);
      f2 = (Pos_PS_R - Fz.Pos);

      Fz.Pos_U_Value[i]=min(f1,f2);

      i++;
    }

    if((Fz.Pos>=Pos_PM_L)&&(Fz.Pos<Pos_PM_R))//论域6
    {
      Fz.Pos_Rule_Value[i]=Pos_PM;

      f1 = (Fz.Pos - Pos_PM_L);
      f2 = (Pos_PM_R - Fz.Pos);

      Fz.Pos_U_Value[i]=min(f1,f2);

      i++;
    }

    if(Fz.Pos >= Pos_PB_L)               //论域7
    {
      Fz.Pos_Rule_Value[i]=Pos_PB;

      f1 = (Fz.Pos - Pos_PB_L);
      f2 = f1;

      Fz.Pos_U_Value[i]=min(f1,f2);

      i++;
    }

    if(i == 1)
    {
       Fz.Pos_U_Value[1]=0;
    }
}


void FuzzyControl::Dpos_fuzzy(void)
{
    int i=0,j=0;
    float f1=0,f2=0;
    for(j=0;j<2;j++)
    {
      Fz.Dpos_Rule_Value[j]=0;
    }

    if(Fz.Dpos < Dpos_NB_R)        //论域1
    {
      Fz.Dpos_Rule_Value[i] = Dpos_NB;  //?????

      f2 = (Dpos_NB_R - Fz.Dpos);
      f1 = f2;

      Fz.Dpos_U_Value[i] = min(f1,f2);

      i++;
    }

    if((Fz.Dpos>=Dpos_NM_L)&&(Fz.Dpos<Dpos_NM_R))//论域2
    {
      Fz.Dpos_Rule_Value[i]=Dpos_NM ;

      f1 = (Fz.Dpos - Dpos_NM_L);
      f2 = (Dpos_NM_R - Fz.Dpos);

      Fz.Dpos_U_Value[i]=min(f1,f2);

      i++;
    }

    if((Fz.Dpos>=Dpos_NS_L)&&(Fz.Dpos<Dpos_NS_R))//论域3
    {
      Fz.Dpos_Rule_Value[i]=Dpos_NS ;

      f1 = (Fz.Dpos - Dpos_NS_L);
      f2 = (Dpos_NS_R - Fz.Dpos);

      Fz.Dpos_U_Value[i]=min(f1,f2);

      i++;
    }


    if((Fz.Dpos>=Dpos_ZO_L)&&(Fz.Dpos<Dpos_ZO_R))//论域4
    {
      Fz.Dpos_Rule_Value[i]=Dpos_ZO;

      f1 = (Fz.Dpos - Dpos_ZO_L);
      f2 = (Dpos_ZO_R - Fz.Dpos);

      Fz.Dpos_U_Value[i]=min(f1,f2);

      i++;
    }

    if((Fz.Dpos>=Dpos_PS_L)&&(Fz.Dpos<Dpos_PS_R))//论域5
    {
      Fz.Dpos_Rule_Value[i]=Dpos_PS ;

      f1 = (Fz.Dpos - Dpos_PS_L);
      f2 = (Dpos_PS_R - Fz.Dpos);

      Fz.Dpos_U_Value[i]=min(f1,f2);

      i++;
    }

    if((Fz.Dpos>=Dpos_PM_L)&&(Fz.Dpos<Dpos_PM_R))//论域6
    {
      Fz.Dpos_Rule_Value[i]=Dpos_PM ;

      f1 = (Fz.Dpos - Dpos_PM_L);
      f2 = (Dpos_PM_R - Fz.Dpos);

      Fz.Dpos_U_Value[i]=min(f1,f2);

      i++;
    }

    if(Fz.Dpos >= Dpos_PB_L)               //论域7
    {
      Fz.Dpos_Rule_Value[i]=Dpos_PB ;

      f1 = (Fz.Dpos - Dpos_PB_L);
      f2 = f1;

      Fz.Dpos_U_Value[i]=min(f1,f2);

      i++;
    }

    if(i==1)
    {
       Fz.Dpos_U_Value[1]=0;
    }
}


float FuzzyControl::Fuzzy_Kp(float servo_Err,float servo_lastErr)
{
     uint8_t i,j;
     Fuzzy_Pos_input(servo_Err);  //对检测到的偏差量化
     Fuzzy_Dpos_input(servo_Err-servo_lastErr);  //对计算出的偏差变化率量化

     Pos_fuzzy();//E模糊化，得模糊值与隶属度
     Dpos_fuzzy();//EC模糊化，得模糊值与隶属度

     Fz.P_Rule[0]=fuzzyRuleKp[Fz.Pos_Rule_Value[0]][Fz.Dpos_Rule_Value[0]];
     Fz.P_Rule[1]=fuzzyRuleKp[Fz.Pos_Rule_Value[1]][Fz.Dpos_Rule_Value[0]];
     Fz.P_Rule[2]=fuzzyRuleKp[Fz.Pos_Rule_Value[0]][Fz.Dpos_Rule_Value[1]];
     Fz.P_Rule[3]=fuzzyRuleKp[Fz.Pos_Rule_Value[1]][Fz.Dpos_Rule_Value[1]];
/* 
     cout<<"Pos_Rule_Value[0]"<<Fz.Pos_Rule_Value[0]<<endl;
     cout<<"Pos_Rule_Value[1]"<<Fz.Pos_Rule_Value[1]<<endl;
     cout<<"Dpos_Rule_Value[0]"<<Fz.Dpos_Rule_Value[0]<<endl;
     cout<<"Dpos_Rule_Value[1]"<<Fz.Dpos_Rule_Value[1]<<endl;  */
     
     Fz.P_U[0] = min(Fz.Dpos_U_Value[0],Fz.Pos_U_Value[0]);
     Fz.P_U[1] = min(Fz.Dpos_U_Value[0],Fz.Pos_U_Value[1]);
     Fz.P_U[2] = min(Fz.Dpos_U_Value[1],Fz.Pos_U_Value[0]);
     Fz.P_U[3] = min(Fz.Dpos_U_Value[1],Fz.Pos_U_Value[1]);

     for(i=0;i<3;i++)
     {
      for(j=i+1;j<4;j++)
      {
        if(Fz.P_Rule[i] == Fz.P_Rule[j])
        {
           if(Fz.P_U[i] > Fz.P_U[j])
           {
             Fz.P_U[j]=0;
           }
           else
           {
             Fz.P_U[i]=0;
           }
        }
      }
     }
     Fz.P_out_u=0;
      Fz.P_out=0;
     /*最大最小法解模糊*/
     /*加权平均法解模糊？*/
      for(i=0;i<4;i++)
      {
        Fz.P_out   += (float)Fz.P_Rule[i]*Fz.P_U[i];
        Fz.P_out_u += (float)(Fz.P_U[i]);
      }

      Fz.P_out =Fz.P_out / Fz.P_out_u;
        
        return Fz.P_out;
}
float  FuzzyControl::Fuzzy_Ki(float servo_Err,float servo_lastErr)
{
     uint8_t i,j;
     Fuzzy_Pos_input(servo_Err);  //对检测到的偏差量化
     Fuzzy_Dpos_input(servo_Err-servo_lastErr);  //对计算出的偏差变化率量化

     Pos_fuzzy();//E模糊化，得模糊值与隶属度
     Dpos_fuzzy();//EC模糊化，得模糊值与隶属度
     
     Fz.I_Rule[0]=fuzzyRuleKi[Fz.Pos_Rule_Value[0]][Fz.Dpos_Rule_Value[0]];
     Fz.I_Rule[1]=fuzzyRuleKi[Fz.Pos_Rule_Value[1]][Fz.Dpos_Rule_Value[0]];
     Fz.I_Rule[2]=fuzzyRuleKi[Fz.Pos_Rule_Value[0]][Fz.Dpos_Rule_Value[1]];
     Fz.I_Rule[3]=fuzzyRuleKi[Fz.Pos_Rule_Value[1]][Fz.Dpos_Rule_Value[1]];
     
     

     Fz.I_U[0] = min(Fz.Dpos_U_Value[0],Fz.Pos_U_Value[0]);
     Fz.I_U[1] = min(Fz.Dpos_U_Value[0],Fz.Pos_U_Value[1]);
     Fz.I_U[2] = min(Fz.Dpos_U_Value[1],Fz.Pos_U_Value[0]);
     Fz.I_U[3] = min(Fz.Dpos_U_Value[1],Fz.Pos_U_Value[1]);

     for(i=0;i<3;i++)
     {
      for(j=i+1;j<4;j++)
      {
        if(Fz.I_Rule[i] == Fz.I_Rule[j])
        {
           if(Fz.I_U[i] > Fz.I_U[j])
           {
             Fz.I_U[j]=0;
           }
           else
           {
             Fz.I_U[i]=0;
           }
        }

      }
     }
     Fz.I_out_u=0;
      Fz.I_out=0;
     /*最大最小法解模糊*/
     /*加权平均法解模糊？*/
      for(i=0;i<4;i++)
      {
        Fz.I_out   += (float)Fz.I_Rule[i]*Fz.I_U[i];
        Fz.I_out_u += (float)(Fz.I_U[i]);
      }

      Fz.I_out =Fz.I_out / Fz.I_out_u;

        return Fz.I_out;
}
float   FuzzyControl::Fuzzy_Kd(float servo_Err,float servo_lastErr)
{
     uint8_t i,j;
     Fuzzy_Pos_input(servo_Err);  //对检测到的偏差量化
     Fuzzy_Dpos_input(servo_Err-servo_lastErr);  //对计算出的偏差变化率量化

     Pos_fuzzy();//E模糊化，得模糊值与隶属度
     Dpos_fuzzy();//EC模糊化，得模糊值与隶属度
    
     Fz.D_Rule[0]=fuzzyRuleKd[Fz.Pos_Rule_Value[0]][Fz.Dpos_Rule_Value[0]];
     Fz.D_Rule[1]=fuzzyRuleKd[Fz.Pos_Rule_Value[1]][Fz.Dpos_Rule_Value[0]];
     Fz.D_Rule[2]=fuzzyRuleKd[Fz.Pos_Rule_Value[0]][Fz.Dpos_Rule_Value[1]];
     Fz.D_Rule[3]=fuzzyRuleKd[Fz.Pos_Rule_Value[1]][Fz.Dpos_Rule_Value[1]];
     
     Fz.D_U[0] = min(Fz.Dpos_U_Value[0],Fz.Pos_U_Value[0]);
     Fz.D_U[1] = min(Fz.Dpos_U_Value[0],Fz.Pos_U_Value[1]);
     Fz.D_U[2] = min(Fz.Dpos_U_Value[1],Fz.Pos_U_Value[0]);
     Fz.D_U[3] = min(Fz.Dpos_U_Value[1],Fz.Pos_U_Value[1]);

     for(i=0;i<3;i++)
     {
      for(j=i+1;j<4;j++)
      {
        if(Fz.D_Rule[i] == Fz.D_Rule[j])
        {
           if(Fz.D_U[i] > Fz.D_U[j])
           {
             Fz.D_U[j]=0;
           }
           else
           {
             Fz.D_U[i]=0;
           }
        }

      }
     }
     Fz.D_out_u=0;
      Fz.D_out=0;
     /*最大最小法解模糊*/
     /*加权平均法解模糊？*/
      for(i=0;i<4;i++)
      {
        Fz.D_out   += (float)Fz.D_Rule[i]*Fz.D_U[i];
        Fz.D_out_u += (float)(Fz.D_U[i]);
      }

      Fz.D_out =Fz.D_out / Fz.D_out_u;

        return Fz.D_out;
}

void FuzzyControl::PID_steering_out(float s_error,float s_derror)
{
   
   if(s_derror>12)
   {
       s_derror=12;
   }
   if(s_derror<-12)
    {
        s_derror=-12;
    }
    

    last_servo_out=steering_KP*s_error+steering_KD*s_derror;
    if(last_servo_out>165)              //输出限幅
        last_servo_out=165;
    if(last_servo_out<-165)
        last_servo_out=-165;
   
   
    
}
 void FuzzyControl::FuzzyPID(float servo_Err,float servo_lastErr)
 {

    //  cout<<"***running_parameter***"<<endl;
      steering_KP=Fuzzy_Kp(servo_Err,servo_lastErr);
      steering_KD=Fuzzy_Kd(servo_Err,servo_lastErr);
     // steering_KD=5;
     // PID_steering_out(m_car.s_error,m_car.s_derror);  
      
   /*  cout<<"m_car.s_error"<<m_car.s_error<<endl;
      cout<<"m_car.s_derror"<<m_car.s_derror<<endl;
      cout<<"steering_KP:"<<steering_KP<<endl;
      cout<<"steering_KD:"<<steering_KD<<endl;
      cout<<"last_servo_out:"<<last_servo_out<<endl;
      cout<<"************************"<<endl;     */
 }
