#include "../include/PurePersuit.hpp"
using namespace std;
using namespace cv;

void PurePersuit::selectAchor(vector<Point2f> & centerLine){

    Point2f temp;
    float d;
    bool select=false;
    for(int i=0;i<centerLine.size();i++){
        temp=computeWorldCoordinate(centerLine[i]);
        d=sqrt((temp.x)*(temp.x)+(temp.y)*(temp.y));

        //如果都比预瞄距离远，那么就去用第一个点
        if(d>aheadDistance&&i==0){
            targetInWorld=temp;
            targetInImage=centerLine[0];
            aheadDistance=d;//被迫更新这一次的预瞄距离
            select=true;
            // cout<<" 1 pure Dis: "<<aheadDistance<<endl;
        }
        if(d>aheadDistance-0.03&&d<aheadDistance+0.03){
            targetInWorld=temp;
            targetInImage=centerLine[i];
            select=true;
            // cout<<"true"<<endl;
            //cout<<"------Find Achor------"<<endl;
            break;
        }
    }
    if(!select){
        targetInWorld=temp;
        targetInImage=centerLine[centerLine.size()-1];
        aheadDistance=d;//被迫更新这一次的预瞄距离
        // cout<<" 2 pure Dis: "<<aheadDistance<<endl;
    }
    // cout<<"pure Dis: "<<aheadDistance<<endl;
    return ;
}

Point2f PurePersuit::computeWorldCoordinate(Point2f& targetInImage){
    Point2f targetInWorld;
    targetInWorld.x=(targetInImage.x-imageCols/2)*metersPerPixel;
    targetInWorld.y=refDistance+(refYonImage-targetInImage.y)*metersPerPixel;
    return targetInWorld;
}

float PurePersuit::computeTheta(){
    theta= atanf(targetInWorld.x/targetInWorld.y)*180/CV_PI;
    return theta;
}

float PurePersuit::ComputeSteeringAngle(){
    steeringAngle=atanf((2*wheelBase*sin(theta*CV_PI/180))/aheadDistance)*180/CV_PI;
    return steeringAngle;
}

void PurePersuit::drawImage(Mat& outputImage){
    //cout<<"--->targetInWorld: "<<targetInWorld<<endl;
    //cout<<"----------->Theta: "<<theta<<endl;
   // cout<<"--->sterringAngle: "<<steeringAngle<<endl;
    circle(outputImage,targetInImage,3,Scalar(255,0,0),-1);
    putText(outputImage,"sterringAngle:"+to_string(steeringAngle),Point(10,180),FONT_HERSHEY_COMPLEX,0.3,Scalar(0,255,0));

}