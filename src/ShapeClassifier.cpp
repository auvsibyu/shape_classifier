#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#define TRIANGLE 0
#define STAR 1
#define SEMICIRCLE 2
#define PENTAGON 3
#define SQUARE 4
#define CROSS 5
#define CIRCLE 6
#define OCTAGON 7
#define HEPTAGON 8
#define HEXAGON 9
#define TRAPEZOID 10
#define RECTANGLE 11
#define QUARTERCIRCLE 12

void classify(double M1, double M2, double M3, double M4);
double checkScore(std::vector<double> shape_M,int shape);
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
                cv::approxPolyDP( contours[i], contours_poly[0], 0.00005*peri, true );
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
            classify(M_matrix[z][0],M_matrix[z][1],M_matrix[z][2],M_matrix[z][3]);
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

void classify(double M1, double M2, double M3, double M4)
{
    // Get Shape Metrics
    std::vector<double> shape_M(4);
    shape_M[0] = M1;
    shape_M[1] = M2;
    shape_M[2] = M3;
    shape_M[3] = M4;

    // Check if triangle
    double tri_score;
    tri_score = checkScore(shape_M,TRIANGLE);
    std::cout << "tri_score is " << tri_score << std::endl;
    // Check if star
    double star_score;
    star_score = checkScore(shape_M,STAR);
    std::cout << "star_score is " << star_score << std::endl;
    // Check if semicircle
    double semicircle_score;
    semicircle_score = checkScore(shape_M,SEMICIRCLE);
    std::cout << "semicircle_score is " << semicircle_score << std::endl;
    // Check if pentagon
    double pentagon_score;
    pentagon_score = checkScore(shape_M,PENTAGON);
    std::cout << "pentagon_score is " << pentagon_score << std::endl;
    // Check if square
    double square_score;
    square_score = checkScore(shape_M,SQUARE);
    std::cout << "square_score is " << square_score << std::endl;
    // Check if cross
    double cross_score;
    cross_score = checkScore(shape_M,CROSS);
    std::cout << "cross_score is " << cross_score << std::endl;
    // Check if circle
    double circle_score;
    circle_score = checkScore(shape_M,CIRCLE);
    std::cout << "circle_score is " << circle_score << std::endl;
    // Check if octagon
    double octagon_score;
    octagon_score = checkScore(shape_M,OCTAGON);
    std::cout << "octagon_score is " << octagon_score << std::endl;
    // Check if heptagon
    double heptagon_score;
    heptagon_score = checkScore(shape_M,HEPTAGON);
    std::cout << "heptagon_score is " << heptagon_score << std::endl;
    // Check if hexagon
    double hexagon_score;
    hexagon_score = checkScore(shape_M,HEXAGON);
    std::cout << "hexagon_score is " << hexagon_score << std::endl;
    // Check if trapezoid
    double trapezoid_score;
    trapezoid_score = checkScore(shape_M,TRAPEZOID);
    std::cout << "trapezoid_score is " << trapezoid_score << std::endl;
    // Check if rectangle
    double rectangle_score;
    rectangle_score = checkScore(shape_M,RECTANGLE);
    std::cout << "rectangle_score is " << rectangle_score << std::endl;
    double quartercircle_score;
    quartercircle_score = checkScore(shape_M,QUARTERCIRCLE);
    std::cout << "quartercircle_score is " << quartercircle_score << std::endl;
    return;
}

double checkScore(std::vector<double> shape_M,int shape)
{
    std::vector<double> ideal_M(4);
    cv::Mat weighting = cv::Mat::zeros(4,4,CV_64F);
    if (shape == TRIANGLE)
    {
        // Define Ideal Triangle Metrics
        ideal_M[0] = 1.02263;
        ideal_M[1] = 20.5368;
        ideal_M[2] = 0.981585;
        ideal_M[3] = 0.980017;

        // Define Weighting for Triangle Metrics
        weighting.at<double>(0,0) = .25;
        weighting.at<double>(1,1) = .25;
        weighting.at<double>(2,2) = .25;
        weighting.at<double>(3,3) = .25;
    }
    else if (shape == STAR)
    {
        // Define Ideal Star Metrics
        ideal_M[0] = 1.24138;
        ideal_M[1] = 14.5534;
        ideal_M[2] = 0.465244;
        ideal_M[3] = 0.805898;

        // Define Weighting for Star Metrics
        weighting.at<double>(0,0) = .25;
        weighting.at<double>(1,1) = .25;
        weighting.at<double>(2,2) = .25;
        weighting.at<double>(3,3) = .25;
    }
    else if (shape == SEMICIRCLE)
    {
        // Define Ideal Semicircle Metrics
        ideal_M[0] = 1.00975;
        ideal_M[1] = 17.4116;
        ideal_M[2] = 0.655672;
        ideal_M[3] = 0.9907;

        // Define Weighting for Semicircle Metrics
        weighting.at<double>(0,0) = .25;
        weighting.at<double>(1,1) = .25;
        weighting.at<double>(2,2) = .25;
        weighting.at<double>(3,3) = .25;
    }
    else if (shape == PENTAGON)
    {
        // Define Ideal Pentagon Metrics
        ideal_M[0] = 1.01184;
        ideal_M[1] = 14.5303;
        ideal_M[2] = 0.460515;
        ideal_M[3] = 0.988823;

        // Define Weighting for Pentagon Metrics
        weighting.at<double>(0,0) = .25;
        weighting.at<double>(1,1) = .25;
        weighting.at<double>(2,2) = .25;
        weighting.at<double>(3,3) = .25;
    }
    else if (shape == SQUARE)
    {
        // Define Ideal Square Metrics
        ideal_M[0] = 1.00362;
        ideal_M[1] = 16.0307;
        ideal_M[2] = 0.535063;
        ideal_M[3] = 0.996447;

        // Define Weighting for Square Metrics
        weighting.at<double>(0,0) = .25;
        weighting.at<double>(1,1) = .25;
        weighting.at<double>(2,2) = .25;
        weighting.at<double>(3,3) = .25;
    }
    else if (shape == CROSS)
    {
        // Define Ideal Cross Metrics
        ideal_M[0] = 1.27732;
        ideal_M[1] = 13.3333;
        ideal_M[2] = 0.431724;
        ideal_M[3] = 0.808809;

        // Define Weighting for Cross Metrics
        weighting.at<double>(0,0) = .25;
        weighting.at<double>(1,1) = .25;
        weighting.at<double>(2,2) = .25;
        weighting.at<double>(3,3) = .25;
    }
    else if (shape == CIRCLE)
    {
        // Define Ideal Circle Metrics
        ideal_M[0] = 1.06349;
        ideal_M[1] = 12.5743;
        ideal_M[2] = 0.4149;
        ideal_M[3] = 0.940341;

        // Define Weighting for Circle Metrics
        weighting.at<double>(0,0) = .25;
        weighting.at<double>(1,1) = .25;
        weighting.at<double>(2,2) = .25;
        weighting.at<double>(3,3) = .25;
    }
    else if (shape == OCTAGON)
    {
        // Define Ideal Octagon Metrics
        ideal_M[0] = 1.06398;
        ideal_M[1] = 13.2208;
        ideal_M[2] = 0.430851;
        ideal_M[3] = 0.940329;

        // Define Weighting for Octagon Metrics
        weighting.at<double>(0,0) = .25;
        weighting.at<double>(1,1) = .25;
        weighting.at<double>(2,2) = .25;
        weighting.at<double>(3,3) = .25;
    }
    else if (shape == HEPTAGON)
    {
        // Define Ideal Heptagon Metrics
        ideal_M[0] = 1.12117;
        ideal_M[1] = 13.3147;
        ideal_M[2] = 0.436272;
        ideal_M[3] = 0.891923;

        // Define Weighting for Heptagon Metrics
        weighting.at<double>(0,0) = .25;
        weighting.at<double>(1,1) = .25;
        weighting.at<double>(2,2) = .25;
        weighting.at<double>(3,3) = .25;
    }
    else if (shape == HEXAGON)
    {
        // Define Ideal Hexagon Metrics
        ideal_M[0] = 1.04896;
        ideal_M[1] = 13.8313;
        ideal_M[2] = 0.498846;
        ideal_M[3] = 0.953324;

        // Define Weighting for Hexagon Metrics
        weighting.at<double>(0,0) = .25;
        weighting.at<double>(1,1) = .25;
        weighting.at<double>(2,2) = .25;
        weighting.at<double>(3,3) = .25;
    }
    else if (shape == TRAPEZOID)
    {
        // Define Ideal Trapezoid Metrics
        ideal_M[0] = 1.04189;
        ideal_M[1] = 17.083;
        ideal_M[2] = 0.686692;
        ideal_M[3] = 0.959795;

        // Define Weighting for Trapezoid Metrics
        weighting.at<double>(0,0) = .25;
        weighting.at<double>(1,1) = .25;
        weighting.at<double>(2,2) = .25;
        weighting.at<double>(3,3) = .25;
    }
    else if (shape == RECTANGLE)
    {
        // Define Ideal Rectangle Metrics
        ideal_M[0] = 1.0;
        ideal_M[1] = 16.6454;
        ideal_M[2] = 0.499647;
        ideal_M[3] = 1.0;

        // Define Weighting for Rectangle Metrics
        weighting.at<double>(0,0) = .25;
        weighting.at<double>(1,1) = .25;
        weighting.at<double>(2,2) = .25;
        weighting.at<double>(3,3) = .25;
    }
    else if (shape == QUARTERCIRCLE)
    {
        // Define Ideal Quarter Circle Metrics
        ideal_M[0] = 1.04535;
        ideal_M[1] = 15.9988;
        ideal_M[2] = 0.625148;
        ideal_M[3] = 0.956621;

        // Define Weighting for Quarter Circle Metrics
        weighting.at<double>(0,0) = .25;
        weighting.at<double>(1,1) = .25;
        weighting.at<double>(2,2) = .25;
        weighting.at<double>(3,3) = .25;
    }
    else
    {
        ideal_M[0] = 1;
        ideal_M[1] = 1;
        ideal_M[2] = 1;
        ideal_M[3] = 1;

        // Define Weighting for Triangle Metrics
        weighting.at<double>(0,0) = .25;
        weighting.at<double>(1,1) = .25;
        weighting.at<double>(2,2) = .25;
        weighting.at<double>(3,3) = .25;
        std::cout << "Incorrect Shape Input in Shape Classifier" << std::endl;
    }

    // Compute Error for each metric
    cv::Mat error = cv::Mat_<double>(4,1);
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

