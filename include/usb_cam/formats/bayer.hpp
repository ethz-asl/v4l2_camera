#ifndef USB_CAM__FORMATS__BAYER_GRBG10_HPP_
#define USB_CAM__FORMATS__BAYER_GRBG10_HPP_

#include "linux/videodev2.h"
#include "usb_cam/formats/pixel_format_base.hpp"
#include "usb_cam/formats/utils.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>

namespace usb_cam
{
namespace formats
{

class BAYER_GRBG10 : public pixel_format_base
{
public:
    explicit BAYER_GRBG10(const format_arguments_t & args = format_arguments_t())
    : pixel_format_base(
        "bayer_grbg10",
        V4L2_PIX_FMT_SGRBG10,
        usb_cam::constants::RGB8,
        3,
        8,
        true), // True indicates that this needs a conversion to RGB8
    _height(args.height),
    _width(args.width),
    _use_cuda(cv::cuda::getCudaEnabledDeviceCount() > 0)
    {
        _rgb8_bytes = 3 * _height * _width;
        _reallocate_images();
    }

    /// @brief Convert a BAYER_GRBG10 image to RGB8
    /// @param src pointer to source BAYER_GRBG10 image
    /// @param dest pointer to destination RGB8 image
    /// @param bytes_used number of bytes used by source image
    void convert(const char* &src, char* &dest, const int& bytes_used) override {
        (void)bytes_used;
        _reallocate_images();
        _bayer_image.data = (uchar*)src;

        if (_use_cuda) {
            cv::_gpu_bayer_image.upload(_bayer_image);
            cv::cuda::demosaicing(_gpu_bayer_image, _gpu_rgb_image, cv::COLOR_BayerGR2RGB);
            _gpu_rgb_image.download(_rgb_image);

        } else {
            cv::demosaicing(_bayer_image, _rgb_image, cv::COLOR_BayerGR2RGB);
        }

        _rgb_image.convertTo(_rgb_image_8bit, CV_8U, _scaling_16_to_8);
        std::memcpy(dest, _rgb_image_8bit.data, _rgb8_bytes);
    }

private:
    bool _use_cuda = false;
    const double _scaling_16_to_8 = 256.0 / 65536.0;
    cv::cuda::GpuMat _gpu_bayer_image;
    cv::cuda::GpuMat _gpu_rgb_image;
    cv::Mat _bayer_image;
    cv::Mat _rgb_image_8bit;
    cv::Mat _rgb_image;
    int _height = 0;
    int _rgb8_bytes = 0;
    int _width = 0;

    void _reallocate_images() {
        // Simplification of the condition: All frames have the same size
        if (_bayer_image.rows != _height || _bayer_image.cols != _width) {
            _bayer_image = cv::Mat(_height, _width, CV_16UC1);
            _rgb_image = cv::Mat(_height, _width, CV_16UC3);
            _rgb_image_8bit = cv::Mat(_height, _width, CV_8UC3);

            if (_use_uda)
                _gpu_bayer_image = cv::cuda::GpuMat(_height, _width, CV_16UC1);
                _gpu_rgb_image = cv::cuda::GpuMat(_height, _width, CV_16UC3);
            }

    }
};

}  // namespace formats
}  // namespace usb_cam

#endif  // USB_CAM__FORMATS__BAYER_GRBG10_HPP_
