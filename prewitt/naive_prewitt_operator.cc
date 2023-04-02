#include "naive_prewitt_operator.h"


NaivePrewittOperator::NaivePrewittOperator(const cv::Mat& grayscale_img):
    kPrewittKernel{-1.0f, 0.0f, 1.0f},
    kPrewittKernelMat(cv::Mat(3, 1, CV_32F, (void*) kPrewittKernel.data())),
    kSumMask{1.0f, 1.0f, 1.0f},
    kSumMat(cv::Mat(3, 1, CV_32F, (void*) kSumMask.data()))
{
    grayscale_img.convertTo(grayscale_img_float_, CV_32F);
    process_horizontal_and_vertical_derivative();
    process_gradient_magnitude();
}


cv::Mat&
NaivePrewittOperator::get_horizontal_derivative(void)
{
    return hor_der_mat_;
}


cv::Mat&
NaivePrewittOperator::get_vertical_derivative(void)
{
    return ver_der_mat_;
}


cv::Mat&
NaivePrewittOperator::get_gradient_magnitude(void)
{
    return grad_mag_mat_;
}


void
NaivePrewittOperator::process_horizontal_and_vertical_derivative(void)
{
    const int kKernelSize = kPrewittKernel.size();
    const int NUM_ROWS = grayscale_img_float_.rows - kKernelSize - 1;
    const int NUM_COLS = grayscale_img_float_.cols - kKernelSize - 1;

    hor_der_mat_f_ = cv::Mat(NUM_ROWS, NUM_COLS, CV_32F, cv::Scalar(0));
    ver_der_mat_f_ = hor_der_mat_f_.clone();

    /**
     * This is a very naive way to run convolution and a function like
     * cv::filter2D is better for performance, but the advantage of this
     * implementation is that is easier to understand the basic algorithm. 
     */
    for(int i = 0; i < NUM_ROWS; i++)
    {
        for(int j = 0; j < NUM_COLS; j++)
        {
            /* ROI stands for region of interest */
            const cv::Rect roi(j, i, kKernelSize, kKernelSize);
            const cv::Mat img_roi = grayscale_img_float_(roi);

            hor_der_mat_f_.at<float>(i, j) = mult_roi_with_hor_kernel(img_roi);
            ver_der_mat_f_.at<float>(i, j) = mult_roi_with_ver_kernel(img_roi);
        }
    }

    hor_der_mat_f_.convertTo(hor_der_mat_, CV_8S);
    ver_der_mat_f_.convertTo(ver_der_mat_, CV_8S);
}


void
NaivePrewittOperator::process_gradient_magnitude(void)
{
    grad_mag_mat_f_ = (hor_der_mat_f_.mul(hor_der_mat_f_)
                       + ver_der_mat_f_.mul(ver_der_mat_f_));

    cv::sqrt(grad_mag_mat_f_, grad_mag_mat_f_);

    grad_mag_mat_f_.convertTo(grad_mag_mat_, CV_8S);
}


float
NaivePrewittOperator::mult_roi_with_hor_kernel(const cv::Mat& roi)
{
    const cv::Mat hor_der = ((kPrewittKernelMat.t() * roi * kSumMat)
                             / kPrewittKernel.size());

    return hor_der.at<float>(0);
}


float
NaivePrewittOperator::mult_roi_with_ver_kernel(const cv::Mat& roi)
{
    const cv::Mat ver_der = ((kSumMat.t() * roi * kPrewittKernelMat)
                             / kPrewittKernel.size());

    return ver_der.at<float>(0);
}
