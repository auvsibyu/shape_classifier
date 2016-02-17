#include "blob_detector/blob_detector.h"

namespace blob_detector {

BlobDetector::BlobDetector() :
    it_(nh_)
{
    //subscriptions
    image_subscriber_ = it_.subscribe("/usb_cam/image_raw", 1, &BlobDetector::imageCallback, this);
    //publications
    image_publisher_ = it_.advertise("/blob_detector/detected_blobs", 1);
}

BlobDetector::~BlobDetector()
{
    //cv::destroyWindow(OPENCV_WINDOW);
}
void BlobDetector::imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    //Convert image to OpenCV format
    try
    {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    // Convert the image to grayscale
    cv::cvtColor( cv_ptr->image, src_gray, CV_BGR2GRAY);

    //Detect Regions Using MSER
    cv::MSER mser(10,200,20000,.5,.2,200,1.01,0.003,5);
    mser(src_gray,contours);

    cv::imshow("WINDOW2", src_gray);

    // Create Mask
    mask = cv::Mat::zeros( src_gray.size(), CV_8UC1 );
    for (int i = 0; i<contours.size(); i++){
        for (int j = 0; j<contours[i].size(); j++){
            p = contours[i][j];
            mask.at<uchar>(p.y, p.x) = 255;
        }
    }

    // Erode and Dilate to remove small regions
    int dilation_size = 2;
    element = getStructuringElement(cv::MORPH_RECT,
                                            cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
                                            cv::Point(dilation_size, dilation_size) );
    dilate(mask,mask,element);
    int erosion_size = 2;
    element = getStructuringElement(cv::MORPH_ELLIPSE,
                                    cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
                                    cv::Point(erosion_size, erosion_size) );
    erode(mask,mask,element);

    sensor_msgs::ImagePtr image_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", mask).toImageMsg();

    image_publisher_.publish(image_msg);

//    // Find morphological gradient to find edges
//    cv::morphologyEx(mask,mask,cv::MORPH_GRADIENT,cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(3,3)));

//    //Find Contours
//    cv::findContours(mask,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);
//    std::vector<std::vector<cv::Point> > contours_poly( contours.size() );
//    cv::Scalar color( 255, 0, 0 );

    cv::imshow("OPENCV_WINDOW", cv_ptr->image);
    cv::imshow("mask",mask);

    cv::waitKey(3);
}
}

// Rejected Code:
//    for(int i = 0; i <contours.size(); i++)
//    {
//        //Approximate Contours with Polygons
//        peri = cv::arcLength(contours[i], true);
//        cv::approxPolyDP( contours[i], contours_poly[i], 0.015*peri, true );
//    }



//    for (int i=0;i< contours_poly.size();i++)
//    {
//        //if there are 4 vertices in the contour(It should be a quadrilateral)
//        if(contours_poly[i].size()==4 )
//        {
//            //iterating through each point
//            cv::Point pt[4];
//            for(int j=0;j<4;j++){
//                pt[j] = contours_poly[i][j];
//            }

//            //drawing lines around the quadrilateral
//            cv::line(cv_ptr->image, pt[0], pt[1], cvScalar(0,255,0),4);
//            cv::line(cv_ptr->image, pt[1], pt[2], cvScalar(0,255,0),4);
//            cv::line(cv_ptr->image, pt[2], pt[3], cvScalar(0,255,0),4);
//            cv::line(cv_ptr->image, pt[3], pt[0], cvScalar(0,255,0),4);
//        }
//    }

//    //Find Contours
//    std::vector<cv::Vec4i> hierarchy;
//    findContours(mask,contours, hierarchy,CV_RETR_TREE,CV_CHAIN_APPROX_SIMPLE);

//    // Convert the image to HSV
//    cv::cvtColor( cv_ptr->image, src_hsv, CV_BGR2HSV);

//    // Split the channels
//    std::vector<cv::Mat> channels;
//    split(src_hsv, channels);
//    cv::Mat hue = channels[0];
//    cv::threshold(channels[1], channels[1], 75, 255, cv::THRESH_TOZERO);

//    //Apply Mask to Image dst
//    cv::Mat dst;
//    src_gray.copyTo(dst, mask);

//    //Find Edges in Image
//    int ratio = 3;
//    int kernel_size = 3;
//    int lowThreshold = 50;
//    cv::Mat detected_edges;
//    /// Reduce noise with a kernel 3x3
//    cv::blur( src_gray, detected_edges, cv::Size(4,4) );

//    /// Canny detector
//    cv::Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );

//    /// Using Canny's output as a mask, we display our result
//    cv::Mat dst2;
//    dst2 = cv::Scalar::all(0);

//    // Apply Mask to Canny's Output
//    mask.copyTo( dst2, detected_edges);



//    // Approximate contours to polygons + get bounding rects and circles

//      std::vector<cv::Rect> boundRect( contours.size() );
//      std::vector<cv::Point2f>center( contours.size() );
//      std::vector<float>radius( contours.size() );

//      for( int i = 0; i < contours.size(); i++ )
//         { cv::approxPolyDP( cv::Mat(contours[i]), contours_poly[i], 1, true );
//           boundRect[i] = boundingRect( cv::Mat(contours_poly[i]) );
//           minEnclosingCircle( (cv::Mat)contours_poly[i], center[i], radius[i] );
//         }

//      /// Draw polygonal contour + bonding rects + circles
//      cv::RNG rng(12345);
//      cv::Mat drawing = cv::Mat::zeros( cv_ptr->image.size(), CV_8UC3 );
//      for( int i = 0; i< contours.size(); i++ )
//         {
//           cv::Scalar color = cv::Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
//           cv::drawContours( drawing, contours_poly, i, color, 1, 8, std::vector<cv::Vec4i>(), 0, cv::Point() );
//           //cv::rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0 );
//           //cv::circle( drawing, center[i], (int)radius[i], color, 2, 8, 0 );
//         }

//      /// Show in a window
//      cv::namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
//      imshow( "Contours", drawing );
