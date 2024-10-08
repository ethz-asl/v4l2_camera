#ifndef USB_CAM__FORMATS__BAYER_GRBG10_HPP_
#define USB_CAM__FORMATS__BAYER_GRBG10_HPP_

#include "linux/videodev2.h"
#include "usb_cam/formats/pixel_format_base.hpp"
#include "usb_cam/formats/utils.hpp"

#include <opencv2/opencv.hpp>

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
    _width(args.width) {
        _rgb8_bytes = 3 * _height * _width;
    }

    /// @brief Convert a BAYER_GRBG10 image to RGB8
    /// @param src pointer to source BAYER_GRBG10 image
    /// @param dest pointer to destination RGB8 image
    /// @param bytes_used number of bytes used by source image
    void convert(const char* &src, char* &dest, const int& bytes_used) override {
        (void)bytes_used;
        const uint16_t* src_16bit = reinterpret_cast<const uint16_t*>(src);
        cv::Mat bayer_image(_height, _width, CV_16UC1, (void*)src_16bit);
        cv::Mat rgb_image;
        cv::demosaicing(bayer_image, rgb_image, cv::COLOR_BayerGR2RGB);
        std::memcpy(dest, rgb_image.data, _rgb8_bytes);

        cv::imshow("Image", rgb_image);
        cv::waitKey(0);
    }

private:
    int _height;
    int _rgb8_bytes;
    int _width;
};

}  // namespace formats
}  // namespace usb_cam

#endif  // USB_CAM__FORMATS__BAYER_GRBG10_HPP_
