#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

#include "naive_prewitt_operator.h"


int
main(int argc, char** argv )
{
    if ( argc != 2 )
    {
        std::cout << "usage: " << argv[0] << " <image path>" << std::endl;
        return -1;
    }
    else
    {
        const cv::Mat image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
        NaivePrewittOperator pw_op(image);
        std::vector<cv::Mat> matrices = {pw_op.get_horizontal_derivative(),
                                         pw_op.get_vertical_derivative(),
                                         pw_op.get_gradient_magnitude()};
        cv::Mat out;

        cv::hconcat(matrices, out);

        cv::imshow("Edge detection with Prewitt Operator", out);
        cv::waitKey(0);

        return 0;
    }
}
