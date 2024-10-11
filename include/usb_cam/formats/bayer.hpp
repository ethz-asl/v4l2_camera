#ifndef USB_CAM__FORMATS__BAYER_GRBG10_HPP_
#define USB_CAM__FORMATS__BAYER_GRBG10_HPP_

#include "linux/videodev2.h"
#include "usb_cam/formats/pixel_format_base.hpp"
#include "usb_cam/formats/utils.hpp"

#include <deque>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xphoto.hpp>

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
    _wb_blue_gain(args.wb_blue_gain),
    _wb_green_gain(args.wb_green_gain),
    _wb_red_gain(args.wb_red_gain),
    _height(args.height),
    _use_cuda(cv::cuda::getCudaEnabledDeviceCount() > 0),
    _width(args.width) {
        _rgb8_bytes = 3 * _height * _width;
        _reallocate_images();
        std::cout << "OpenCV version: " << CV_VERSION << std::endl;
        if (_use_cuda) {
            std::cout << "Using CUDA: " << _use_cuda << std::endl;
            cv::cuda::setDevice(0);
        }
        // TODO: Only do on CPU
        _wb = cv::xphoto::createGrayworldWB();
    }

    /// @brief Convert a BAYER_GRBG10 image to RGB8
    /// @param src pointer to source BAYER_GRBG10 image
    /// @param dest pointer to destination RGB8 image
    /// @param bytes_used number of bytes used by source image
    void convert(const char* &src, char* &dest, const int& bytes_used) override {
        (void)bytes_used;
        // https://www.kernel.org/doc/html/v4.9/media/uapi/v4l/pixfmt-srggb10.html
        // In 10 bit Bayer GRBG format
        // Each row has 2 * width pixels
        // Only reallocate if the size changes
        _reallocate_images();
        _bayer_image.data = (uchar*)src;

        // Demosaic and convert to 8 bit mat
        // RVIZ and/or ROS expects BGR
        if (_use_cuda) {
            _gpu_bayer_image.upload(_bayer_image);
            cv::cuda::cvtColor(_gpu_bayer_image, _gpu_rgb_image, cv::COLOR_BayerGR2BGR);
            _gpu_rgb_image.download(_rgb_image);

        } else {
            cv::cvtColor(_bayer_image, _rgb_image, cv::COLOR_BayerGR2BGR);
        }

        // Directly shift the 16-bit pixel values to 8-bit
        // TODO: Casting on GPU? Would be possible on opencv 4.10.0
        for (int y = 0; y < _rgb_image.rows; ++y) {
            const cv::Vec3w* row_in = _rgb_image.ptr<cv::Vec3w>(y);
            cv::Vec3b* row_out = _rgb_image_8bit.ptr<cv::Vec3b>(y);
            for (int x = 0; x < _rgb_image.cols; ++x) {
                row_out[x][0] = static_cast<uchar>((row_in[x][0] >> 8));
                row_out[x][1] = static_cast<uchar>((row_in[x][1] >> 8));
                row_out[x][2] = static_cast<uchar>((row_in[x][2] >> 8));
            }
        }

        // Apply gains from launch file if specified
        // otherwise runs auto calibration grayworldWB
        _gray_world_wb(_rgb_image_8bit, _rgb_image_out);

        // TODO: If casting and wb is done on gpu
        // if (_use_cuda && false) {
        //     _gpu_rgb_image_8bit.download(_rgb_image_8bit);
        // }
        std::memcpy(dest, _rgb_image_out.data, _rgb8_bytes);
    }

private:
    bool _use_cuda = false;
    const int _wb_N = 100;
    cv::cuda::GpuMat _gpu_bayer_image;
    cv::cuda::GpuMat _gpu_rgb_image_8bit;
    cv::cuda::GpuMat _gpu_rgb_image;
    cv::Mat _bayer_image;
    cv::Mat _rgb_image_8bit;
    cv::Mat _rgb_image_out;
    cv::Mat _rgb_image;
    cv::Ptr<cv::xphoto::WhiteBalancer> _wb;
    float _wb_blue_gain;
    float _wb_green_gain;
    float _wb_red_gain;
    int _height = 0;
    int _rgb8_bytes = 0;
    int _width = 0;
    std::deque<float> _wb_blue_gains;
    std::deque<float> _wb_green_gains;
    std::deque<float> _wb_red_gains;

    void _reallocate_images() {
        if (_bayer_image.rows != _height || _bayer_image.cols != _width) {
            _bayer_image = cv::Mat(_height, _width, CV_16UC1);
            _rgb_image = cv::Mat(_height, _width, CV_16UC3);
            _rgb_image_8bit = cv::Mat(_height, _width, CV_8UC3);

            if (_use_cuda) {
                _gpu_bayer_image = cv::cuda::GpuMat(_height, _width, CV_16UC1);
                _gpu_rgb_image = cv::cuda::GpuMat(_height, _width, CV_16UC3);
                _gpu_rgb_image_8bit = cv::cuda::GpuMat(_height, _width, CV_8UC3);
            }
        }
    }

    void _gray_world_wb(cv::Mat& in, cv::Mat& out) {
        std::vector<cv::Mat> channels;
        cv::split(in, channels);
        float avg_blue_gain = 0.0f;
        float avg_green_gain = 0.0f;
        float avg_red_gain = 0.0f;

        if (_wb_blue_gain < 1e-6f || _wb_green_gain < 1e-6f || _wb_red_gain < 1e-6f) {
            float means[3];
            float mean_gw;

            for (size_t i=0; i < 3; i++) {
                means[i] = cv::mean(channels[i])[0];
                mean_gw += means[i];
            }
            mean_gw /= 3.0;

            const float wb_blue_gain = mean_gw / means[0];
            const float wb_green_gain = mean_gw / means[1];
            const float wb_red_gain = mean_gw / means[2];

            // Add the new gains to the deque and maintain window size N
            if (_wb_blue_gains.size() >= _wb_N) _wb_blue_gains.pop_front();  // Remove oldest if at limit
            if (_wb_green_gains.size() >= _wb_N) _wb_green_gains.pop_front();
            if (_wb_red_gains.size() >= _wb_N) _wb_red_gains.pop_front();

            _wb_blue_gains.push_back(wb_blue_gain);
            _wb_green_gains.push_back(wb_green_gain);
            _wb_red_gains.push_back(wb_red_gain);

            // Compute the average gains over the window
            avg_blue_gain = _compute_average(_wb_blue_gains);
            avg_green_gain = _compute_average(_wb_green_gains);
            avg_red_gain = _compute_average(_wb_red_gains);
            std::cout << avg_blue_gain << ", " << avg_green_gain << ", " << avg_red_gain << std::endl;

        } else {
            avg_blue_gain = _wb_blue_gain;
            avg_green_gain = _wb_green_gain;
            avg_red_gain = _wb_red_gain;
        }

        // Apply the averaged gains
        channels[0] *= avg_blue_gain;
        channels[1] *= avg_green_gain;
        channels[2] *= avg_red_gain;
        cv::merge(channels, out);
    }

    float _compute_average(const std::deque<float>& gains) {
        float sum = 0.0f;
        for (float gain : gains) {
            sum += gain;
        }
        return sum / gains.size();
}
};

}  // namespace formats
}  // namespace usb_cam

#endif  // USB_CAM__FORMATS__BAYER_GRBG10_HPP_
