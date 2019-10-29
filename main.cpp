#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    VideoCapture cap(0);
    if(!cap.isOpened())return -1;

    int minm=120,maxm=300;
    for(;;)
    {
        Mat frame,frame2,dst,cdst;
        cap >> frame;
        cap >> frame2;

        namedWindow("input image", 1);
        namedWindow("filter lines", 1);
        namedWindow("find lines", 1);
        namedWindow("final image", 1);



        createTrackbar("min", "filter lines", &minm, 200);
        createTrackbar("max", "filter lines", &maxm, 500);


        Canny(frame, dst, 50, 200, 3);
        cvtColor(dst, cdst, CV_GRAY2BGR);
        blur(dst, dst, Size(5 , 5));
        threshold(dst, dst, 0, 255, THRESH_OTSU);


        vector<Vec4i> lines;
        HoughLinesP(dst, lines, 1, CV_PI/180, minm, maxm, 3000 );

         for( size_t i = 0; i < lines.size(); i++ )
          {
            Vec4i l = lines[i];
            line( cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 5, CV_AA);
            line( frame2, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 5, CV_AA);
          }


        imshow("input image", frame);
        imshow("filter lines", dst);
        imshow("find lines", cdst);
        imshow("final image", frame2);
        imwrite("detected.jpg", frame2);


        if(waitKey(10) >= 0) break;   // you can increase delay to 2 seconds here
    }
 return 0;
}
