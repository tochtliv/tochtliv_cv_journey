#pragma once

#include <opencv2/opencv.hpp>


class NaivePrewittOperator
{
    public:
        NaivePrewittOperator(const cv::Mat& grayscale_img);

        cv::Mat& get_horizontal_derivative(void);

        cv::Mat& get_vertical_derivative(void);

        cv::Mat& get_gradient_magnitude(void);

    private:
        const std::vector<float> kPrewittKernel;

        const std::vector<float> kSumMask;

        const cv::Mat kPrewittKernelMat;

        const cv::Mat kSumMat;

        cv::Mat grayscale_img_float_;

        cv::Mat hor_der_mat_;

        cv::Mat ver_der_mat_;

        cv::Mat grad_mag_mat_;

        cv::Mat hor_der_mat_f_;

        cv::Mat ver_der_mat_f_;

        cv::Mat grad_mag_mat_f_;

        void process_horizontal_and_vertical_derivative(void);

        void process_gradient_magnitude(void);

        float mult_roi_with_hor_kernel(const cv::Mat& roi);

        float mult_roi_with_ver_kernel(const cv::Mat& roi);
};
