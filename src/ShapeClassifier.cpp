#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream>

double triArea(int A,int B,int C, std::vector<cv::Point> points);
void maxAreaTriangle(std::vector<cv::Point> &points, std::vector<cv::Point2f> &maxTri);

int main(int argc, char** argv )
{
    if ( argc != 3 )
    {
        printf("usage: Picture Directory <Directory> <Number of Images>\n");
        return -1;
    }

    std::string directory = argv[1];
    std::string st1 = "file";
    std::string ext = ".jpg";
    std::string filename;
    int count;
    std::istringstream convert(argv[2]);
    if (!(convert >> count))
    {
        count = 0;
    }

    int z;

    std::vector<std::vector<double> > M_matrix(count+1,std::vector<double>(4));

    for(z = 1; z <= count; z++)
    {
        std::stringstream ss;
        ss << z;
        filename = directory + "/" + st1 + ss.str() + ext;

        cv::Mat image;
        image = cv::imread( filename, 1 );

        if ( !image.data )
        {
            printf("No image data \n");
            return -1;
        }

        cv::Mat src_gray;

        // Convert the image to grayscale
        cv::cvtColor( image, src_gray, CV_BGR2GRAY);

        std::vector<std::vector<cv::Point> > contours;

        //Detect Regions Using MSER
        cv::MSER mser(10,100000,1500000,.5,.2,200,1.01,0.003,5);
        mser(src_gray,contours);

//        cv::Mat mask;
//        cv::Point p;
//        cv::Mat element;
//        // Create Mask
//        mask = cv::Mat::zeros( src_gray.size(), CV_8UC1 );
//        for (int i = 0; i<contours.size(); i++){
//            for (int j = 0; j<contours[i].size(); j++){
//                p = contours[i][j];
//                mask.at<uchar>(p.y, p.x) = 255;
//            }
//        }

//        // Erode and Dilate to remove small regions
//        int dilation_size = 2;
//        element = getStructuringElement(cv::MORPH_RECT,
//                                        cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
//                                        cv::Point(dilation_size, dilation_size) );
//        dilate(mask,mask,element);
//        int erosion_size = 2;
//        element = getStructuringElement(cv::MORPH_ELLIPSE,
//                                        cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
//                                        cv::Point(erosion_size, erosion_size) );
//        erode(mask,mask,element);

//        // Find morphological gradient to find edges
//        cv::morphologyEx(mask,mask,cv::MORPH_GRADIENT,cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(3,3)));

        //Find Contours
//        cv::findContours(mask,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);
        std::vector<std::vector<cv::Point> > contours_poly( 1 );
        std::vector<cv::Point> current_contour;
        double max_contour_area = -1.0;
        double current_contour_area;
        //Approximate Contours with Polygons and pull off largest region only
        double peri;

        for(int i = 0; i <contours.size(); i++)
        {
            current_contour_area = cv::contourArea(contours[i], false);
            if (current_contour_area>max_contour_area && !isnan(current_contour_area))
            {
                cv::approxPolyDP( contours[i], contours_poly[0], 0.005*peri, true );
                peri = cv::arcLength(contours[i], true);
            }
        }

        std::vector<double> Acp(contours_poly.size());
        for( int i = 0; i < contours_poly.size(); i++ )
        {
            Acp[i] = cv::contourArea(contours_poly[i]);
        }


        // Find the convex hull object for each contour
        std::vector<std::vector<cv::Point> >hull( contours_poly.size() );
        for( int i = 0; i < contours_poly.size(); i++ )
        {
            cv::convexHull( cv::Mat(contours_poly[i]), hull[i], false );
        }

        // Find minimum-area enclosing rectangle
        std::vector<cv::RotatedRect> minRect( contours_poly.size() );
        std::vector<double> Abb(contours_poly.size());

        for( int i = 0; i < contours_poly.size(); i++ )
        {
            minRect[i] = cv::minAreaRect( cv::Mat(contours_poly[i]) );
            Abb[i] = minRect[i].size.width * minRect[i].size.height;
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
                maxAreaTriangle(hull[i],maxTri[i]);
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
        //Metric #5: Acp/Abb

        for (int i = 0; i < contours_poly.size(); i++)
        {
            M_matrix[z][0] = len[i]/Pch[i];
            M_matrix[z][1] = pow(Pch[i],2.0)/Ach[i];
            M_matrix[z][2] = Alt[i]/Ach[i];
            M_matrix[z][3] = Pch[i]/len[i];
            M_matrix[z][4] = Acp[i]/Abb[i];
        }

        //Draw Output on Image
        // Convert the image to color

        // Choose Colors
        cv::Scalar red( 255, 0, 0 );
        cv::Scalar green( 0, 255, 0 );
        cv::Scalar blue( 0, 0, 255);
        cv::Scalar cyan(255, 255, 0);
        cv::Scalar yellow(0,255,255);
        for(int i = 0; i < contours_poly.size(); i++)
        {
            //Contours
            cv::drawContours(image,contours_poly,i,red,10);
            cv::drawContours(image, hull, i, green, 10, 8, std::vector<cv::Vec4i>(), 0, cv::Point() );

            //Minimum-Area Enclosing Rectangle
            for( int j = 0; j < 4; j++ )
            {
                cv::Point2f rect_points[4];
                minRect[i].points( rect_points );
                cv::line(image, rect_points[j], rect_points[(j+1)%4], blue, 10, 8 );
            }
            //Max-Area enclosed triangle
            if (hull[i].size() > 2)
            {
                for( int j = 0; j < 3; j++ )
                {

                    cv::line(image, maxTri[i][j], maxTri[i][(j+1)%3], cyan, 10, 8);
                }
            }
        }

        cv::namedWindow("WINDOW" + ss.str(), cv::WINDOW_NORMAL);
        cv::resizeWindow("WINDOW" + ss.str(), 640,480);
        cv::imshow("WINDOW" + ss.str(), image);
        std::cout << "Image " << ss.str() << std::endl;
        std::cout << "M1 (len/Pch) is " << M_matrix[z][0] << std::endl;
        std::cout << "M2 (Pch^2/Ach) is " << M_matrix[z][1] << std::endl;
        std::cout << "M3 (Alt/Ach) is " << M_matrix[z][2] << std::endl;
        std::cout << "M4 (Pch/Per) is " << M_matrix[z][3] << std::endl;
        std::cout << "M5 (Acp/Abb) is " << M_matrix[z][4] << std::endl << std::endl;
    }

    std::cout << "Average" << std::endl;

    for (int i = 0; i<5; i++)
    {
        double sum = 0;
        double average;
        for (int z = 1; z <=count; z++)
        {
            sum = sum + M_matrix[z][i];
        }
        average = sum/count;
        std::stringstream ss;
        ss << i+1;
        std::cout << "Average M" << ss.str() << " is " << average << std::endl;
    }
    cv::waitKey(0);

    return 0;
} 

void maxAreaTriangle(std::vector<cv::Point> &points, std::vector<cv::Point2f> &maxTri)
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
            while (triArea(A,B,C,points) <= triArea(A,B,(C+1)%n,points)) //loop C
            {
                C = (C+1)%n;
            }
            if (triArea(A,B,C,points) <= triArea(A,(B+1)%n,C,points))
            {
                B=(B+1)%n;
                continue;
            }
            else
            {
                break;
            }
        }


        if (triArea(A,B,C,points) > triArea(bA,bB,bC,points))
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


double triArea(int A,int B,int C, std::vector<cv::Point> points)
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

