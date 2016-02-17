#ifndef BLOB_DETECTOR_BLOB_DETECTOR_H
#define BLOB_DETECTOR_BLOB_DETECTOR_H

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <vector>
#include "opencv2/features2d/features2d.hpp"

namespace blob_detector {

//static const std::string OPENCV_WINDOW = "Image window";

class BlobDetector
{
public:
    BlobDetector();
    ~BlobDetector();

private:

    ros::NodeHandle nh_;
    ros::NodeHandle nh_private_;

    image_transport::ImageTransport it_;
    image_transport::Subscriber image_subscriber_;
    image_transport::Publisher image_publisher_;

    cv::Mat src_gray;
    cv::Mat mask;
    cv::Mat element;
    std::vector<std::vector<cv::Point> > contours;
    cv::Point p;
    cv_bridge::CvImagePtr cv_ptr;
    double peri;



    void imageCallback(const sensor_msgs::ImageConstPtr& msg);
};

}

#endif // BLOB_DETECTOR_BLOB_DETECTOR_H

