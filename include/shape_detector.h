#ifndef SHAPE_DETECTOR_SHAPE_DETECTOR_H
#define SHAPE_DETECTOR_SHAPE_DETECTOR_H

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <vector>
#include "opencv2/features2d/features2d.hpp"

namespace shape_detector {

class ShapeDetector
{
public:
    ShapeDetector();
    ~ShapeDetector();

private:

    ros::NodeHandle nh_;
    ros::NodeHandle nh_private_;

    image_transport::ImageTransport it_;
    image_transport::Subscriber image_subscriber_;
    image_transport::Publisher image_publisher_;

    cv::Mat src_color;
    std::vector<std::vector<cv::Point> > contours;
    cv_bridge::CvImagePtr cv_ptr;
    double peri;
    void imageCallback(const sensor_msgs::ImageConstPtr& msg);
    double triArea(int A,int B,int C, std::vector<cv::Point> points);
    void maxAreaTriangle(std::vector<cv::Point> &points,std::vector<cv::Point2f> &maxTri);

};

}

#endif // SHAPE_DETECTOR_SHAPE_DETECTOR_H
