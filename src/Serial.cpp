#include "Serial.hpp"
#include <fcntl.h>
#include <sys/select.h>
#include <string.h>
#include <unistd.h>
#include <iostream>

Serial::Serial() {
}
 
Serial::~Serial() {
}

//串口初始化
int Serial::Serial_init(const char *  path)
{
    serialport = std::make_shared<SerialPort>();
    if (serialport == nullptr) {
      std::cerr << "Serial Create Failed ." << std::endl;
      return -1;
    }
    // try检测语句块有没有异常
    try 
    {
      serialport->Open(path);                     // 打开串口
      serialport->SetBaudRate(BaudRate::BAUD_115200); // 设置波特率
      serialport->SetCharacterSize(CharacterSize::CHAR_SIZE_8); // 8位数据位
      serialport->SetFlowControl(FlowControl::FLOW_CONTROL_NONE); // 设置流控
      serialport->SetParity(Parity::PARITY_NONE);                 // 偶校验
      serialport->SetStopBits(StopBits::STOP_BITS_1); // 1个停止位
    } 
    catch (const OpenFailed &) // catch捕获并处理 try 检测到的异常。
    {
      std::cerr << "Serial port: " << path << "open failed ..."
                << std::endl;
      isOpen = false;
      return -2;
    } 
    catch (const AlreadyOpen &) // catch捕获并处理 try 检测到的异常。
    {
      std::cerr << "Serial port: " << path << "open failed ..."
                << std::endl;
      isOpen = false;
      return -3;
    }
     catch (...) // catch捕获并处理 try 检测到的异常。
    {
      std::cerr << "Serial port: " << path << " recv exception ..."
                << std::endl;
      isOpen = false;
      return -4;
    }
    isOpen = true;
    return 0;
}

int Serial::transmitByte(unsigned char data) 
{
    // try检测语句块有没有异常
    try {
      serialport->WriteByte(data); // 写数据到串口
    } catch (const std::runtime_error &) // catch捕获并处理 try 检测到的异常。
    {
      std::cerr << "The Write() runtime_error." << std::endl;
      return -2;
    } catch (const NotOpen &) // catch捕获并处理 try 检测到的异常。
    {
      std::cerr << "Port Not Open ..." << std::endl;
      return -1;
    }
    serialport->DrainWriteBuffer(); // 等待，直到写缓冲区耗尽，然后返回。
    return 0;
}

int Serial::Serial_write(float err, size_t len)
{
    if (!isOpen || serialport == nullptr) {
      return -1;
    }
    //设置舵机值
    setError(err);
    captureLastTxFrame(len);
    ////整包发送数据
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
    //   serialport->DrainWriteBuffer(); //等待，直到写缓冲区耗尽，然后返回。--会加大耗时 旧板卡5ms左右
    // serialport.Write(send_buffer);
    // serialport.DrainWriteBuffer();
    return 0;
}

  int Serial::receiveBytes(unsigned char &charBuffer, size_t msTimeout = 0) 
  {
    /*try检测语句块有没有异常。如果没有发生异常,就检测不到。
    如果发生异常，則交给 catch 处理，执行 catch 中的语句* */
    if (!isOpen || serialport == nullptr) {
      return -1;
    }
    try {
      /*从串口读取一个数据,指定msTimeout时长内,没有收到数据，抛出异常。
      如果msTimeout为0，则该方法将阻塞，直到数据可用为止。*/
      serialport->ReadByte(charBuffer, msTimeout); // 可能出现异常的代码段
    }
     catch (const ReadTimeout &) // catch捕获并处理 try 检测到的异常。
    {
      // std::cerr << "The ReadByte() call has timed out." << std::endl;
      return -2;
    } catch (const NotOpen &) // catch()中指明了当前 catch 可以处理的异常类型
    {
      std::cerr << "Port Not Open ..." << std::endl;
      return -1;
    } catch (const std::runtime_error &) {
      std::cerr << "The ReadByte() runtime_error." << std::endl;
      return -3;
    }
    return 0;
  }

int Serial::openUart(const char * path) {
    firstFlag=0;
    fd = open(path, O_RDWR);
    if (fd == -1) {
       std::cout<<"open uart fail."<<std::endl;
       return -1;
    }
    
    return 1;
}
 
int Serial::setUart(int _speed) {
    /*配置串口参数*/
    speed_t speed;
    
    memset(&myios, 0, sizeof (myios));
    //设置串口工作在本地模式
    myios.c_cflag |= CLOCAL | CREAD;
    cfmakeraw(&myios);
    //波特率
    switch (_speed) {
        case 9600:
            speed = B9600;
            break;
        case 19200:
            speed = B19200;
            break;
        case 115200:
            speed = B115200;
            break;
        default:
            return -1;
    }
    cfsetispeed(&myios, speed);
    cfsetospeed(&myios, speed);
    //8位数据位
    myios.c_cflag &= ~CSIZE; /* 用数据位掩码清空数据位设置 */
    myios.c_cflag |= CS8;
    //无奇偶校验
    myios.c_cflag &= ~PARENB;
    //一个停止位
    myios.c_cflag &= ~CSTOPB; //一个停止位
    //设置最小等待时间和最少字符数量
    myios.c_cc[VTIME] = 0;
    myios.c_cc[VMIN] = 1;
    //刷新串口
    tcflush(fd, TCIOFLUSH);
    //配置生效
    tcsetattr(fd, TCSANOW, &myios);
    
    return 0;
}
 
int Serial::writeData(int len) {
    int value, ret;
    unsigned char temp[7];
 
    //清空串口数据缓存
    do {
        value = readData(temp, 7, 0, 0);
    } while (value);
    ret = write(fd, send_buffer, len);
    return ret;
}
 
int Serial::readData(void * buf, int len, int sec, int msec) {
    int ret= -1;
    fd_set read_set;
    struct timeval tv;
 
    FD_ZERO(&read_set);
    FD_SET(fd, &read_set);
    tv.tv_sec = sec;
    tv.tv_usec = msec * 1000;
    ret = select(fd + 1, &read_set, 0, 0, &tv);
    switch (ret){ 
        case 0:
            break;
        case -1:
            break;
        default:
            if (FD_ISSET(fd, &read_set)) {
                ret = read(fd, buf, len);
            }
            break;
    }
 
    return ret;
}
 
int Serial::closeUart() {
    return close(fd);
}
