#include "shape_detector/shape_detector.h"
#include <iostream>

namespace shape_detector {

ShapeDetector::ShapeDetector() :
    it_(nh_)
{
    //subscriptions
    image_subscriber_ = it_.subscribe("/blob_detector/detected_blobs", 1, &ShapeDetector::imageCallback, this);
    //publications
    image_publisher_ = it_.advertise("/shape_detector/detected_shapes", 1);
    cv::namedWindow("OPENCV_WINDOW");
}

ShapeDetector::~ShapeDetector()
{
    cv::destroyWindow("OPENCV_WINDOW");
}
void ShapeDetector::imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    //Convert image to OpenCV format
    try
    {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::MONO8);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    // Find morphological gradient to find edges
    cv::morphologyEx(cv_ptr->image,cv_ptr->image,cv::MORPH_GRADIENT,cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(3,3)));

    cv::imshow("WINDOW", cv_ptr->image);

    //Find Contours
    cv::findContours(cv_ptr->image,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);
    std::vector<std::vector<cv::Point> > contours_poly( contours.size() );

    //Approximate Contours with Polygons
    for(int i = 0; i <contours.size(); i++)
    {
        peri = cv::arcLength(contours[i], true);
        cv::approxPolyDP( contours[i], contours_poly[i], 0.015*peri, true );
    }

    // Find the convex hull object for each contour
    std::vector<std::vector<cv::Point> >hull( contours_poly.size() );
    for( int i = 0; i < contours_poly.size(); i++ )
    {
        cv::convexHull( cv::Mat(contours_poly[i]), hull[i], false );
    }

    // Find minimum-area enclosing rectangle
    std::vector<cv::RotatedRect> minRect( contours_poly.size() );

    for( int i = 0; i < contours_poly.size(); i++ )
    {
        minRect[i] = cv::minAreaRect( cv::Mat(contours_poly[i]) );
    }

    // Find arcLength
    std::vector<double> len(contours_poly.size());
    for(int i = 0; i <contours_poly.size(); i++)
    {
        len[i] = cv::arcLength(contours_poly[i], true);
    }

    // Perimeter of Convex Hull
    std::vector<double> Pch(contours_poly.size());
    for(int i = 0; i <contours_poly.size(); i++)
    {
        Pch[i] = cv::arcLength(hull[i], true);
    }

    //Area of Convex Hull
    std::vector<double> Ach(contours_poly.size());
    for(int i = 0; i <contours_poly.size(); i++)
    {
        Ach[i] = cv::contourArea(hull[i],false);
    }

    // Find maximum-area enclosed triangle
    std::vector<std::vector<cv::Point2f> > maxTri(contours_poly.size(),std::vector<cv::Point2f>(3));
    for( int i = 0; i < contours_poly.size(); i++ )
    {
        if (hull[i].size() > 2)
        {
            ShapeDetector::maxAreaTriangle(hull[i],maxTri[i]);
        }
    }

    std::vector<double> Alt(contours_poly.size());
    for(int i = 0; i <contours_poly.size(); i++)
    {
        Alt[i] = cv::contourArea(maxTri[i],false);
    }

    //Metric #1: len/Pch - Close to unity for closed shapes
    //Metric #2: Pch^2/Ach - Thinness Ratio - Small for circles
    //Metric #3: Alt/Ach - Near unity for triangles
    //Metric #4: Pch/Per - Distinguish rectangles from ellipses and diamonds
    std::vector<std::vector<double> > M_matrix(contours_poly.size(),std::vector<double>(4));
    for (int i = 0; i < contours_poly.size(); i++)
    {
        M_matrix[i][1] = len[i]/Pch[i];
        M_matrix[i][2] = pow(Pch[i],2.0)/Ach[i];
        M_matrix[i][3] = Alt[i]/Ach[i];
        M_matrix[i][4] = Pch[i]/len[i];
    }

    //Draw Output on Image
    // Convert the image to color
    cv::cvtColor( cv_ptr->image, src_color, 8);

    // Choose Colors
    cv::Scalar red( 255, 0, 0 );
    cv::Scalar green( 0, 255, 0 );
    cv::Scalar blue( 0, 0, 255);
    cv::Scalar cyan(255, 255, 0);
    cv::Scalar yellow(0,255,255);
    for(int i = 0; i < contours_poly.size(); i++)
    {
        //Contours
        cv::drawContours(src_color,contours_poly,i,red,2);
        cv::drawContours(src_color, hull, i, green, 2, 8, std::vector<cv::Vec4i>(), 0, cv::Point() );

        //Minimum-Area Enclosing Rectangle
        cv::Point2f rect_points[4];
        minRect[i].points( rect_points );
        for( int j = 0; j < 4; j++ )
        {
            cv::line(src_color, rect_points[j], rect_points[(j+1)%4], blue, 2, 8 );
        }
        //Max-Area enclosed triangle
        if (hull[i].size() > 2)
        {
            for( int j = 0; j < 3; j++ )
            {

                cv::line(src_color, maxTri[i][j], maxTri[i][(j+1)%3], cyan, 2, 8);
            }
        }
    }

    cv::imshow("OPENCV_WINDOW", src_color);

    cv::waitKey(3);

    // Publish Image
    sensor_msgs::ImagePtr image_msg = cv_bridge::CvImage(std_msgs::Header(), "rgb8", src_color).toImageMsg();

    image_publisher_.publish(image_msg);
}

void ShapeDetector::maxAreaTriangle(std::vector<cv::Point> &points, std::vector<cv::Point2f> &maxTri)
{
if (points.size()<3)
{
    std::cout << "error" << std::endl;
}
// Assume points have been sorted already, as 0...(n-1)
int A,B,C,bA,bB,bC,iterA,iterB,iterC, n;
n = points.size();
iterA = 0;
iterB = 0;
iterC = 0;
//Current Point
A = 0; B = 1; C = 2;
//The "best" triple of points
bA= A; bB= B; bC= C;
triArea(0,1,2,points);
while (true) //loop A
{
    iterA = iterA+1;
    while (true) //loop B
    {
        iterB = iterB + 1;
        while (ShapeDetector::triArea(A,B,C,points) <= ShapeDetector::triArea(A,B,(C+1)%n,points)) //loop C
        {
            C = (C+1)%n;
        }
        if (ShapeDetector::triArea(A,B,C,points) <= ShapeDetector::triArea(A,(B+1)%n,C,points))
        {
            B=(B+1)%n;
            continue;
        }
        else
        {
            break;
        }
    }

    if (ShapeDetector::triArea(A,B,C,points) > ShapeDetector::triArea(bA,bB,bC,points))
    {
        bA = A; bB = B; bC = C;
    }

    A = (A+1)%n;
    if (A==B)
    {
        B=(B+1)%n;
    }
    if (B==C)
    {
        C=(C+1)%n;
    }
    if (A==0)
    {
        break;
    }
}
maxTri[0] = points[bA];
maxTri[1] = points[bB];
maxTri[2] = points[bC];
}

double ShapeDetector::triArea(int A,int B,int C, std::vector<cv::Point> points)
    {
    std::vector<cv::Point> tri_points(3);
    tri_points[0].x = points[A].x;
    tri_points[0].y = points[A].y;
    tri_points[1].x = points[B].x;
    tri_points[1].y = points[B].y;
    tri_points[2].x = points[C].x;
    tri_points[2].y = points[C].y;
    return contourArea(tri_points);
    }
}
