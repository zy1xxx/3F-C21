#include "opencv2/opencv.hpp"
#include <vector>
#include <math.h>
#include "driver.h"
using namespace cv;
using namespace std;
int diff;
double slideWindow(Mat canny_img){
    bool showFlag=true;
    
    int Pheight = canny_img.rows;//图片宽高
    int Pwidth = canny_img.cols;
    int width = 150;//滑动窗宽高
    int height = 20;
    int linewidthMin = 20;//检测点之间的最小距离
    Point2i originPoint = Point2i(Pwidth / 2 - width / 2, Pheight - height);//初始框原点
    Mat canny_img2 = canny_img.clone();//canny_img2为显示的图层
	vector<Point2i> recpointls;
    vector<Point2i>::iterator recpointlsP;//recpointls的迭代器
    double originAngle=0;
    //确定第一个滑框的位置
    
    for(int leftRightCtn=0;leftRightCtn<3;leftRightCtn++){
        int position = 0;
        int cnt = 0;
        int poSum = 0;
        bool line1 = false;
        int restart = 0;
        for(int i=0;i<width;i++){
            if(line1 == false){//开始检测第一根线
                try{
                    if((int)canny_img.at<uchar>(originPoint.y, originPoint.x + i) == 255){//如果有黑色的像素点
                        cnt++;
                        poSum = poSum + i;
                        restart = i + linewidthMin;//下一个开始位置
                        line1 = true;
					}
				}
                catch(std::exception& e){
                    cout<<"something wrong";
				}
			}
            else{
                if (i < restart){
                    continue;
				}
                else{
                    try{//检测第二根线
                        if ((int)canny_img.at<uchar>(originPoint.y, originPoint.x + i) == 255){
                            cnt++;
                            poSum = poSum + i;
                            break;
						}
					}
                    catch(std::exception& e){
                        cout<<"something wrong";
					}
				}
			}
		}
        if(cnt==0){//说明线偏移中线
            if (leftRightCtn==0){//先往右偏移
                originPoint.x=originPoint.x+width;
                originAngle=0;
			}
            else if (leftRightCtn==1){
                originPoint.x=originPoint.x-2*width;
                originAngle=0;
			}
            else{
                //cout<<"line is too far\n";
                originPoint = Point2i(Pwidth / 2 - width / 2, Pheight - height);//初始框原点
                position=width / 2;
                Point2i newPoint = Point2i(int(originPoint.x+ position - width / 2), originPoint.y - height);//下一个滑动窗的位置
                recpointls.push_back(originPoint);
                cv::rectangle(canny_img2, originPoint, Point2i(originPoint.x + width, originPoint.y + height),
                            (255, 0, 255), 2);//画出滑动窗
                originPoint = newPoint;//迭代
                //exit(0)
			}
        }
        else{
            position = poSum / cnt;//中线的位置
            Point2i newPoint = Point2i(int(originPoint.x+ position - width / 2), originPoint.y - height);//下一个滑动窗的位置
            recpointls.push_back(originPoint);
            cv::rectangle(canny_img2, originPoint, Point2i(originPoint.x + width, originPoint.y + height),
                        (255, 0, 255), 2);//画出滑动窗
            originPoint = newPoint;//迭代
            break;
		}
	}

    for(int v=0;v<8;v++){
        int position = 0;
        int cnt = 0;
        int poSum = 0;
        bool line1 = false;
        int restart = 0;
        for(int i=0;i<width;i++){
            if(line1 == false){//开始检测第一根线
                try{
                    if((int)canny_img.at<uchar>(originPoint.y, originPoint.x + i) == 255){//如果有黑色的像素点
                        cnt++;
                        poSum = poSum + i;
                        restart = i + linewidthMin;//下一个开始位置
                        line1 = true;
					}
				}
                catch(std::exception& e){
                    cout<<"something wrong";
				}
			}
            else{
                if (i < restart){
                    continue;
				}
                else{
                    try{//检测第二根线
                        if ((int)canny_img.at<uchar>(originPoint.y, originPoint.x + i) == 255){
                            cnt++;
                            poSum = poSum + i;
                            break;
						}
					}
                    catch(std::exception& e){
                        cout<<"something wrong";
					}
				}
			}
		}
        if (cnt != 0){
            position = poSum / cnt;
		}
        else{
            position=width / 2;
        }
        Point2i newPoint = Point2i(int(originPoint.x+ position - width / 2), originPoint.y - height);//下一个滑动窗的位置
            recpointls.push_back(originPoint);
            cv::rectangle(canny_img2, originPoint, Point2i(originPoint.x + width, originPoint.y + height),
                        (255, 0, 255), 2);//画出滑动窗
            originPoint = newPoint;//迭代
    }
    int i=1;
    int len=6;
    recpointlsP=recpointls.begin();
    double slopeRate=double((*(recpointlsP+i+len)).x - (*(recpointlsP+i)).x) / double((*(recpointlsP+i)).y - (*(recpointlsP+i+len)).y);//斜率
    double angle = atan(slopeRate);//弧度
    double angleJ=angle/3.1415*180;//角度
    diff=(*(recpointlsP++)).x+width/2-300;
    cout<<"angleJ"<<angleJ<<endl;
    
    if (showFlag){
        cv::imshow("canny_img", canny_img2);
        }
    //frameCtn+=1
    return angleJ+originAngle;
}
Mat canny(Mat img){
	cv::Mat gray_img;
	cv::cvtColor(img,gray_img,cv::COLOR_BGR2GRAY);
    
    //Mat gray = cv::cvtColor(img,cv::COLOR_RGB2GRAY);   //要二值化图像，要先进行灰度化处理
    //gray=cv::GaussianBlur(gray,cv::Size2d(5,5),0);
    //ret, binary = cv::threshold(gray,0,255,cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::Mat canny_img;
	cv::Canny(gray_img,canny_img,100, 150, 3);
    
    return canny_img;
}
Mat PerspectiveTransfer(Mat img){
	int ROTATED_SIZE  = 600 ;
	cv::Mat out;
	cv::Point2f src_points[] = { 
	cv::Point2f(224, 220),
	cv::Point2f(380, 220),
	cv::Point2f(550, 400),
	cv::Point2f(70, 400) };

	cv::Point2f dst_points[] = {
	cv::Point2f(0, 0),
	cv::Point2f(ROTATED_SIZE, 0),
	cv::Point2f(ROTATED_SIZE, ROTATED_SIZE),
	cv::Point2f(0, ROTATED_SIZE) };
 
	cv::Mat M = cv::getPerspectiveTransform(src_points, dst_points);
	cv::warpPerspective(img,out,M,cv::Size2d(ROTATED_SIZE,ROTATED_SIZE));
	return out;
}
void control(double angle1,double angle2,double sum,int &VL,int &VR){
    double Kp=1.2; //1
    double Ki=0; //0
    double Kd=5; //3 2.5
    double Kc=0.2;
    int V=100;
    double w=Kp*angle1+Ki*sum+Kd*(angle2-angle1)+Kc*diff;
    sum=sum+angle2;
    VR=w/2+V;
    VL=V-w/2;
}
int main(int argc, char** argv)
{
    //读取视频或摄像头
	VideoCapture capture(1);
    Driver car;
    car.initDriver();
    Mat frame;
    capture >> frame;
    double angle1=slideWindow(frame);
    double angleSum=angle1;
    double angle2=angle1;
    int VL,VR;
	while (true)
	{
		capture >> frame;
		frame=PerspectiveTransfer(frame);
		frame=canny(frame);
        angle2=slideWindow(frame);
        angleSum=angleSum+angle2;
        control(angle1,angle2,angleSum,VL,VR);
        angle1=angle2;
        car.set_speed(VL,VR);
		waitKey(30);
	}
	return 0;
}
