#ifndef SERIAL_H
#define SERIAL_H
 
#include     <array>
#include     <string>
// #include <termios.h>
#include     <stdio.h>      /*标准输入输出定义*/
#include     <stdlib.h>     /*标准函数库定义*/
 
#include     <unistd.h>     /*Unix 标准函数定义*/
#include     <sys/types.h> 
#include     <sys/stat.h>
#include     <fcntl.h>      /*文件控制定义*/
#include     <termios.h>    /*PPSIX 终端控制定义*/
#include     <errno.h>      /*错误号定义*/
#include     <stdint.h>
#include     <iostream>
#include     <math.h>
#include     <libserial/SerialPort.h> // 串口通信
#include <chrono>
#include <thread>

using std::string;
using namespace std;
using namespace LibSerial;

class Serial {
 
public:
    Serial();
    virtual ~Serial();
    bool firstFlag = true;
    // float speedSet=1.0;
    float speed=1.0;
    bool gyroFlag=false;//陀螺仪达到某个角度
    bool depotFlag=false;
    bool blind; //致盲
    int blindCount=0;
    bool startFlag = false;

    int Serial_init(const char *  path);
    int transmitByte(unsigned char data);
    int Serial_write(float err, size_t len);
    int receiveBytes(unsigned char &charBuffer, size_t msTimeout);

    int openUart(const char *  path);
    int setUart(int speed);
    int writeData(int len);
    int readData(void * buf, int len, int sec, int msec);
    int closeUart();
    std::array<uint8_t, 11> lastTxFrame() const {
        return last_tx_frame_;
    }
    int64_t lastTxEncodedMonoUs() const {
        return last_tx_encoded_mono_us_;
    }
    void setError(float floatNum){
        
        error.FloatNum=floatNum;
        send_buffer[0]=0xAA;
        send_buffer[1]=0x0B; //整个数据帧的长度 
        for(int i=0;i<4;i++){
            send_buffer[i+2]=error.charNum[i];
        }
        if(speed<0)send_buffer[6]=char('n');
        if(speed>=0)send_buffer[6]=char('p');

        send_buffer[7]=char(fabs(speed)*50); //speed
        send_buffer[8]=char('n');//正常下不用陀螺仪归零
        for(int i=0;i<9;i++)
        {
            sc+=send_buffer[i];
            ac+=sc;
        }
        send_buffer[9]=char(sc);
        send_buffer[10]=char(ac);
        sc=0;
        ac=0;
        // firstFlag=1;
    };
    void setErrorWithSpeed(float floatNum, float forceSpeed){
        // 重载方法：支持强制设置速度（用于启动延迟）
        error.FloatNum=floatNum;
        send_buffer[0]=0xAA;
        send_buffer[1]=0x0B; //整个数据帧的长度 
        for(int i=0;i<4;i++){
            send_buffer[i+2]=error.charNum[i];
        }
        // forceSpeed < 0 表示后退，>= 0 表示前进
        if(forceSpeed < 0) send_buffer[6]=char('n');
        else send_buffer[6]=char('p');

        send_buffer[7]=char(fabs(forceSpeed)*50); //speed
        send_buffer[8]=char('n');//正常下不用陀螺仪归零
        for(int i=0;i<9;i++)
        {
            sc+=send_buffer[i];
            ac+=sc;
        }
        send_buffer[9]=char(sc);
        send_buffer[10]=char(ac);
        sc=0;
        ac=0;
    };
    int Serial_writeWithSpeed(float err, float speedForce, size_t len) {
        // 支持强制速度的串口写入（用于启动延迟等场景）
        if (!isOpen || serialport == nullptr) {
            return -1;
        }
        setErrorWithSpeed(err, speedForce);
        captureLastTxFrame(len);
        try {
            const std::string frame(send_buffer, len);
            serialport->Write(frame);
            serialport->DrainWriteBuffer();
        } catch (const NotOpen &) {
            std::cerr << "Port Not Open ..." << std::endl;
            return -1;
        } catch (const std::runtime_error &) {
            std::cerr << "The Write() runtime_error." << std::endl;
            return -2;
        }
        return 0;
    };
    void stopCar(){

        Serial_write(0,11);
        cout<<"-->stop car <--"<<endl;
    };
    void reversecar()
    {
        cout<<"-->reverse car<--"<<endl;
        Serial_write(0,11);
        this->speed = -0.6;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        Serial_write(100,11);
        this->speed = -0.6;
         std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    //陀螺仪归零校正
    void zeroCorrection(char type){
        if(type=='g'){
            gyroFlag=false;
            send_buffer[6]=char('g');   
            cout<<"发送g"<<endl;
        }
        
        if(type=='f'){
            depotFlag=false;
            send_buffer[6]=char('f');
            cout<<"发送f"<<endl;

        }
        writeData(7);
        return ;
    }
    
    
private:
    void captureLastTxFrame(size_t len) {
        last_tx_frame_.fill(0);
        const size_t copy_len =
            (len < last_tx_frame_.size()) ? len : last_tx_frame_.size();
        for (size_t i = 0; i < copy_len; ++i) {
            last_tx_frame_[i] = static_cast<uint8_t>(
                static_cast<unsigned char>(send_buffer[i]));
        }
        last_tx_encoded_mono_us_ =
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    string path;
    int fd;
    struct termios  myios;
    
    uint8_t sc=0;
    uint8_t ac=0;  
    std::shared_ptr<SerialPort> serialport = nullptr;
    bool isOpen = false;

    typedef union{
        char charNum[4];
        float FloatNum;
    }DataUnion;
    DataUnion error;
    char send_buffer[64];
    char recv_buffer[64];
    std::array<uint8_t, 11> last_tx_frame_{};
    int64_t last_tx_encoded_mono_us_ = 0;
   
};


#endif /* SERIAL_H */
