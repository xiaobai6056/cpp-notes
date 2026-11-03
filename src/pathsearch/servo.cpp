#include "../include/servo.hpp"

using namespace std;


float servo::servoControl(float err,float alpha)
{
    servo_lastErr=servo_Err;
    servo_Err=err;
    float servo_position;
    float pdservo_position;
    float totalservo_position;
    float servo_derror = servo_Err - servo_lastErr;
    // outfile<<"--->fuzzyer.servo_derror="<<servo_derror<<endl;
    if(servo_derror >60) servo_derror= 60;
    if(servo_derror<-60) servo_derror=-60;
    
    if(FuzzyEnable)
    {
        fuzzyer.FuzzyPID(servo_Err,servo_lastErr);
        

        totalservo_position=fuzzyer.steering_KP*servo_Err +fuzzyer.steering_KD*servo_derror;
    }
    else if(PolyEnable)
    {

        servo_position = P3base*pow(servo_Err,3)+Pbase *servo_Err +D3base * pow(servo_derror,3) +Dbase * servo_derror;
        pdservo_position= PdP3base*pow(servo_Err,3)+PdPbase *servo_Err +PdD3base * pow(servo_derror,3) +PdDbase * servo_derror;
        totalservo_position=(1-alpha)*servo_position+alpha*pdservo_position;
    }
    else
    {        
        totalservo_position = Pbase *servo_Err + Dbase * servo_derror;
    }
	
	// Kp=fuzzyer.steering_KP;
	// Kd=fuzzyer.steering_KD;
    //outfile<<servo_Err<<endl;
    // float Kp=Pbase+weight*servo_Err*servo_Err;
    // if(Kp>40)
    //     Kp=40;
    // outfile<<"--->fuzzyer.steering_KP="<<fuzzyer.steering_KP<<endl;
    // outfile<<"--->fuzzyer.steering_KD="<<fuzzyer.steering_KD<<endl;
    
    
    // outfile<<"--->fuzzyer.position="<<servo_position<<endl;
    // float servo_position = Pbase*servo_Err+Dbase*servo_derror;
    // cout<<"当前PD"<<Pbase<<" , "<<Dbase<<endl;
    return totalservo_position;
}