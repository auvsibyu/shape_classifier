#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#define TRIANGLE 0
#define STAR 1
#define SEMICIRCLE 2
#define PENTAGON 3
#define CROSS 5
#define CIRCLE 6
#define OCTAGON 7
#define HEPTAGON 8
#define HEXAGON 9
#define TRAPEZOID 10
#define RECTANGLE 11
#define QUARTERCIRCLE 12

void classify(double M1, double M2, double M3, double M4, double M5, double convex_score,  double width, double height, std::vector<cv::Point>& contour);
double checkScore(std::vector<double> shape_M,int shape);
double triArea(int A,int B,int C, std::vector<cv::Point> points);
void maxAreaTriangle(std::vector<cv::Point> &points, std::vector<cv::Point2f> &maxTri);
static double angle(cv::Point pt1, cv::Point pt2, cv::Point pt0);

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

    std::vector<std::vector<double> > M_matrix(count+1,std::vector<double>(6));

    for(z = 1; z <= count; z++)
    {

        std::stringstream ss;
        ss << z;
        filename = directory + "/" + st1 + ss.str() + ext;

        cv::Mat image;
        image = cv::imread( filename, 1 );

        std::cout << "Image " << ss.str() << std::endl;

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

        cv::Mat mask;
        cv::Point p;
        cv::Mat element;
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

        // Find morphological gradient to find edges
        cv::morphologyEx(mask,mask,cv::MORPH_GRADIENT,cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(3,3)));

        //Find Contours
        cv::findContours(mask,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);
        std::vector<std::vector<cv::Point> > contours_poly( 1 );
        std::vector<std::vector<cv::Point> > rough_contours_poly( 1 );
        double max_contour_area = -1.0;
        double current_contour_area;
        //Approximate Contours with Polygons and pull off largest region only
        double peri;

        for(int i = 0; i <contours.size(); i++)
        {
            current_contour_area = cv::contourArea(contours[i], false);
            if (current_contour_area>max_contour_area && !isnan(current_contour_area))
            {
                peri = cv::arcLength(contours[i], true);
                cv::approxPolyDP( contours[i], contours_poly[0], 0.0001*peri, true );
                cv::approxPolyDP( contours[i], rough_contours_poly[0], .01*peri, true );

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
        //Metric #5: Ach^2/(Abb*Alt)

        for (int i = 0; i < contours_poly.size(); i++)
        {
            M_matrix[z][5] = Ach[i]/Acp[i];
            M_matrix[z][0] = len[i]/Pch[i];
            M_matrix[z][1] = pow(Pch[i],2.0)/Ach[i];
            M_matrix[z][2] = Alt[i]/Ach[i];
            M_matrix[z][3] = Acp[i]/Abb[i];
            M_matrix[z][4] = pow(Ach[i],2.0)/(Abb[i]*Alt[i]);
            classify(M_matrix[z][0],M_matrix[z][1],M_matrix[z][2],M_matrix[z][3],M_matrix[z][4],M_matrix[z][5],minRect[i].size.width,minRect[i].size.height, rough_contours_poly[i]);
        }

//        //Draw Output on Image
//        // Convert the image to color

//        // Choose Colors
//        cv::Scalar red( 255, 0, 0 );
//        cv::Scalar green( 0, 255, 0 );
//        cv::Scalar blue( 0, 0, 255);
//        cv::Scalar cyan(255, 255, 0);
//        cv::Scalar yellow(0,255,255);
//        for(int i = 0; i < contours_poly.size(); i++)
//        {
//            //Contours
//            cv::drawContours(image,contours_poly,i,red,10);
//            cv::drawContours(image, hull, i, green, 10, 8, std::vector<cv::Vec4i>(), 0, cv::Point() );

//            //Minimum-Area Enclosing Rectangle
//            for( int j = 0; j < 4; j++ )
//            {
//                cv::Point2f rect_points[4];
//                minRect[i].points( rect_points );
//                cv::line(image, rect_points[j], rect_points[(j+1)%4], blue, 10, 8 );
//            }
//            //Max-Area enclosed triangle
//            if (hull[i].size() > 2)
//            {
//                for( int j = 0; j < 3; j++ )
//                {

//                    cv::line(image, maxTri[i][j], maxTri[i][(j+1)%3], cyan, 10, 8);
//                }
//            }
//        }

//        cv::namedWindow("WINDOW" + ss.str(), cv::WINDOW_NORMAL);
//        cv::resizeWindow("WINDOW" + ss.str(), 640,480);
//        cv::imshow("WINDOW" + ss.str(), image);

//        std::cout << "M1 (len/Pch) is " << M_matrix[z][0] << std::endl;
//        std::cout << "M2 (Pch^2/Ach) is " << M_matrix[z][1] << std::endl;
//        std::cout << "M3 (Alt/Ach) is " << M_matrix[z][2] << std::endl;
//        std::cout << "M4 (Acp/Abb) is " << M_matrix[z][3] << std::endl;
//        std::cout << "M5 (Ach^2/(Abb*Alt) is " << M_matrix[z][4] << std::endl << std::endl;
    }

//    std::cout << "Average" << std::endl;

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
//        std::cout << "Average M" << ss.str() << " is " << average << std::endl;
    }
    cv::waitKey(0);

    return 0;
} 

void classify(double M1, double M2, double M3, double M4, double M5, double convex_score, double width, double length, std::vector<cv::Point> &contour)
{
    // Get Shape Metrics
    std::vector<double> shape_M(5);
    shape_M[0] = M1;
    shape_M[1] = M2;
    shape_M[2] = M3;
    shape_M[3] = M4;
    shape_M[4] = M5;

    //Shape Decision Implementation
    // Check if Convex
    double convex_lower_bound = 0.90;
    double convex_upper_bound = 1.10;
    if ((convex_lower_bound < convex_score) && (convex_score < convex_upper_bound))
    {
//        std::cout << "Shape is Convex" << std::endl;
        // Check if circle
        double circle_score;
        circle_score = checkScore(shape_M,CIRCLE);
        // Check if semicircle
        double semicircle_score;
        semicircle_score = checkScore(shape_M,SEMICIRCLE);
        //Check if quarter-circle
        double quartercircle_score;
        quartercircle_score = checkScore(shape_M,QUARTERCIRCLE);
        // Check if triangle
        double tri_score;
        tri_score = checkScore(shape_M,TRIANGLE);
        // Check if rectangle
        double rectangle_score;
        rectangle_score = checkScore(shape_M,RECTANGLE);
        // Check if pentagon
        double pentagon_score;
        pentagon_score = checkScore(shape_M,PENTAGON);
        // Check if hexagon
        double hexagon_score;
        hexagon_score = checkScore(shape_M,HEXAGON);
        // Check if heptagon
        double heptagon_score;
        heptagon_score = checkScore(shape_M,HEPTAGON);
        // Check if octagon
        double octagon_score;
        octagon_score = checkScore(shape_M,OCTAGON);

        if (rectangle_score > .8)
        {
            double square_upper_bound = 1.25;
            double square_lower_bound = 0.75;
            // Check if square
            if ((square_lower_bound < length/width) && (length/width < square_upper_bound))
            {
                std::cout << "Target is square" << std::endl;
            }
            else
            {
                std::cout << "Target is rectangle" << std::endl;
            }
        }
        if (circle_score > .95)
        {
            std::cout << "Target is circle" << std::endl;
        }
        if ((semicircle_score > .90) || (quartercircle_score > .90))
        {
            if ((.75 < length/width) && (length/width < 1.25))
            {
                std::cout << "Target is quartercircle" << std::endl;
            }
            else
            {
                std::cout << "Target is semicircle" << std::endl;
            }
        }
        if (tri_score > .90)
        {
            std::cout << "Target is triangle" << std::endl;
        }
        if (pentagon_score > .95)
        {
            std::cout << "Target is pentagon" << std::endl;
        }
        if (hexagon_score > .95)
        {
            std::cout << "Target is hexagon" << std::endl;
        }
        if (heptagon_score > .90)
        {
            std::cout << "Target is heptagon" << std::endl;
        }
        if (octagon_score > .95)
        {
            std::cout << "Target is octagon" << std::endl;
        }
        int vtc = contour.size();
        if (vtc==4)
        {
            std::vector<double> angles;
            for (int j = 2; j <= (vtc+1); j++)
            {
                angles.push_back(angle(contour[j%vtc], contour[(j-2)%vtc] , contour[(j-1)%vtc]));
            }
            std::sort(angles.begin(),angles.end());
            if ((angles[0]-.175 < angles[1]) && (angles[1] < angles[0] + .175))
            {
                if ((angles[2]-.175 < angles[3]) && (angles[3] < angles[2] + .175))
                {
                    if ((angles[1]-.175 > angles[2]) || (angles[2] > angles[1] + .175))
                    {
                        std::cout << "Target is a trapezoid" << std::endl;
                    }
                }
            }
        }
    }
    else
    {
//        std::cout << "Shape is not Convex" << std::endl;
        // Check if star
        double star_score;
        star_score = checkScore(shape_M,STAR);
        if (star_score > .95)
        {
            std::cout << "Target is star" << std::endl;
        }

        int vtc = contour.size();
        if (vtc==12)
        {
            std::vector<double> angles;
            for (int j = 2; j <= (vtc+1); j++)
            {
                angles.push_back(angle(contour[j%vtc], contour[(j-2)%vtc] , contour[(j-1)%vtc]));
            }
            std::sort(angles.begin(),angles.end());
            bool cross = true;
            for (int i = 0; i < vtc; i++)
            {
                if ((1.5708-.175 > angles[i]) || (angles[i] > 1.5708 + .175))
                {
                    cross = false;
                }
            }
            if (cross == true)
            {
                std::cout << "Target is a cross" << std::endl;
            }
        }
    }

//    std::cout << "Vertices: " << vtc << std::endl;
//    std::cout << "rectangle_score is " << rectangle_score << std::endl;
//    std::cout << "circle_score is " << circle_score << std::endl;
//    std::cout << "semicircle_score is " << semicircle_score << std::endl;
//    std::cout << "quartercircle_score is " << quartercircle_score << std::endl;
//    std::cout << "tri_score is " << tri_score << std::endl;
//    std::cout << "pentagon_score is " << pentagon_score << std::endl;
//    std::cout << "hexagon_score is " << hexagon_score << std::endl;
//    std::cout << "heptagon_score is " << heptagon_score << std::endl;
//    std::cout << "octagon_score is " << octagon_score << std::endl;
//    std::cout << "star_score is " << star_score << std::endl;

    return;
}

static double angle(cv::Point pt1, cv::Point pt2, cv::Point pt0)
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return acos((dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10));
}
double checkScore(std::vector<double> shape_M,int shape)
{
    std::vector<double> ideal_M(5);
    cv::Mat weighting = cv::Mat::zeros(5,5,CV_64F);

    if (shape == QUARTERCIRCLE)
    {
        // Define Ideal Quarter Circle Metrics
        ideal_M[0] = 1.04535;
        ideal_M[1] = 15.9988;
        ideal_M[2] = 0.625148;
        ideal_M[3] = 0.778313;
        ideal_M[4] = 1.26394;

        // Define Weighting for Quarter Circle Metrics
        weighting.at<double>(0,0) = 0;
        weighting.at<double>(1,1) = .4;
        weighting.at<double>(2,2) = .1;
        weighting.at<double>(3,3) = .1;
        weighting.at<double>(4,4) = .4;

    }
    else if (shape == SEMICIRCLE)
    {
        // Define Ideal Semicircle Metrics
        ideal_M[0] = 1.00551;
        ideal_M[1] = 17.0925;
        ideal_M[2] = 0.64162;
        ideal_M[3] = 0.770088;
        ideal_M[4] = 1.20784;

        // Define Weighting for Semicircle Metrics
        weighting.at<double>(0,0) = 0;
        weighting.at<double>(1,1) = .4;
        weighting.at<double>(2,2) = .1;
        weighting.at<double>(3,3) = .1;
        weighting.at<double>(4,4) = .4;
    }
    else if (shape == CIRCLE)
    {
        // Define Ideal Circle Metrics
        ideal_M[0] = 1.06349;
        ideal_M[1] = 12.5743;
        ideal_M[2] = 0.4149;
        ideal_M[3] = 0.799082;
        ideal_M[4] = 1.89937;

        // Define Weighting for Circle Metrics
        weighting.at<double>(0,0) = 0;
        weighting.at<double>(1,1) = .4;
        weighting.at<double>(2,2) = .2;
        weighting.at<double>(3,3) = .2;
        weighting.at<double>(4,4) = .2;
    }
    else if (shape == TRIANGLE)
    {
        // Define Ideal Triangle Metrics
        ideal_M[0] = 0.0001;
        ideal_M[1] = 0.0001;
        ideal_M[2] = 0.0001;
        ideal_M[3] = 0.5;
        ideal_M[4] = 0.0001;

        // Define Weighting for Triangle Metrics
        weighting.at<double>(3,3) = 1;
    }
    else if (shape == RECTANGLE)
    {
        // Define Ideal Rectangle Metrics
        ideal_M[0] = 1.0;
        ideal_M[1] = 16.0307;
        ideal_M[2] = 0.5;
        ideal_M[3] = 1.0;
        ideal_M[4] = 2.;

        // Define Weighting for Rectangle Metrics
        weighting.at<double>(0,0) = 0;
        weighting.at<double>(1,1) = .3;
        weighting.at<double>(2,2) = .3;
        weighting.at<double>(3,3) = .2;
        weighting.at<double>(4,4) = .2;
    }
    else if (shape == PENTAGON)
    {
        // Define Ideal Pentagon Metrics
        ideal_M[0] = 1.01184;
        ideal_M[1] = 14.5303;
        ideal_M[2] = 0.460515;
        ideal_M[3] = 0.706979;
        ideal_M[4] = 1.55087;

        // Define Weighting for Pentagon Metrics
        weighting.at<double>(0,0) = 0;
        weighting.at<double>(1,1) = .3;
        weighting.at<double>(2,2) = .3;
        weighting.at<double>(3,3) = .2;
        weighting.at<double>(4,4) = .2;
    }
    else if (shape == HEXAGON)
    {
        // Define Ideal Hexagon Metrics
        ideal_M[0] = 1.04896;
        ideal_M[1] = 13.8313;
        ideal_M[2] = 0.498846;
        ideal_M[3] = 0.750278;
        ideal_M[4] = 1.50626;

        // Define Weighting for Hexagon Metrics
        weighting.at<double>(0,0) = 0;
        weighting.at<double>(1,1) = .3;
        weighting.at<double>(2,2) = .3;
        weighting.at<double>(3,3) = .2;
        weighting.at<double>(4,4) = .2;
    }
    else if (shape == HEPTAGON)
    {
        // Define Ideal Heptagon Metrics
        ideal_M[0] = 1.12117;
        ideal_M[1] = 13.3147;
        ideal_M[2] = 0.436272;
        ideal_M[3] = 0.735172;
        ideal_M[4] = 1.71839;

        // Define Weighting for Heptagon Metrics
        weighting.at<double>(0,0) = 0;
        weighting.at<double>(1,1) = .2;
        weighting.at<double>(2,2) = .2;
        weighting.at<double>(3,3) = .3;
        weighting.at<double>(4,4) = .3;
    }
    else if (shape == OCTAGON)
    {
        // Define Ideal Octagon Metrics
        ideal_M[0] = 1.06398;
        ideal_M[1] = 13.2208;
        ideal_M[2] = 0.430851;
        ideal_M[3] = 0.819734;
        ideal_M[4] = 1.90586;

        // Define Weighting for Octagon Metrics
        weighting.at<double>(0,0) = 0;
        weighting.at<double>(1,1) = .3;
        weighting.at<double>(2,2) = .1;
        weighting.at<double>(3,3) = .3;
        weighting.at<double>(4,4) = .3;
    }
    else if (shape == STAR)
    {
        // Define Ideal Star Metrics
        ideal_M[0] = 1.24138;
        ideal_M[1] = 14.5534;
        ideal_M[2] = 0.465244;
        ideal_M[3] = 0.340881;
        ideal_M[4] = 1.54815;

        // Define Weighting for Star Metrics
        weighting.at<double>(0,0) = .2;
        weighting.at<double>(1,1) = .2;
        weighting.at<double>(2,2) = .1;
        weighting.at<double>(3,3) = .3;
        weighting.at<double>(4,4) = .2;
    }
    else
    {
        ideal_M[0] = 1;
        ideal_M[1] = 1;
        ideal_M[2] = 1;
        ideal_M[3] = 1;
        ideal_M[4] = 1;

        // Default Weighting
        weighting.at<double>(0,0) = .1;
        weighting.at<double>(1,1) = .3;
        weighting.at<double>(2,2) = .2;
        weighting.at<double>(3,3) = .1;
        weighting.at<double>(4,4) = .3;

        std::cout << "Incorrect Shape Input in Shape Classifier" << std::endl;
    }

    // Compute Error for each metric
    cv::Mat error = cv::Mat_<double>(5,1);
    for (int i = 0; i < ideal_M.size(); i++)
    {
        error.at<double>(i,0) = 1.0 - fabs((shape_M[i]-ideal_M[i])/ideal_M[i]);
    }

    // Get transpose of error
    cv::Mat error_trans;
    cv::transpose(error,error_trans);

    // Compute Score
    double score;
    score = cv::Mat(error_trans*weighting*error).at<double>(0,0);
    return score;
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

