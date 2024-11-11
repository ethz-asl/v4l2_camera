#include "ros/console.h"
#include <deque>
#include <Eigen/Dense>
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
        true),
    _wb_gains(args.wb_blue_gain, args.wb_green_gain, args.wb_red_gain),
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

        _wb = cv::xphoto::createGrayworldWB();
        _use_grayworld_wb = (_wb_gains[0] < 1e-6f || _wb_gains[1] < 1e-6f || _wb_gains[2] < 1e-6f);
    }

    void convert(const char* &src, char* &dest, const int& bytes_used) override {
        (void)bytes_used;
        _reallocate_images();
        _bayer_image.data = (uchar*)src;

        if (_use_cuda && !_use_grayworld_wb) {
            _gpu_bayer_image.upload(_bayer_image);
            cv::cuda::cvtColor(_gpu_bayer_image, _gpu_rgb_image, cv::COLOR_BayerGR2BGR);
            _apply_white_balance_gpu();
            _gpu_rgb_image_8bit.download(_rgb_image_out);

        } else {
            cv::cvtColor(_bayer_image, _rgb_image, cv::COLOR_BayerGR2BGR);
            _apply_white_balance_cpu();
        }

        std::memcpy(dest, _rgb_image_out.data, _rgb8_bytes);
    }

private:
    bool _use_cuda;
    bool _use_grayworld_wb;
    const int _wb_N = 100;
    cv::cuda::GpuMat _gpu_bayer_image;
    cv::cuda::GpuMat _gpu_rgb_image_8bit;
    cv::cuda::GpuMat _gpu_rgb_image;
    cv::Mat _bayer_image;
    cv::Mat _rgb_image_8bit;
    cv::Mat _rgb_image_out;
    cv::Mat _rgb_image;
    cv::Ptr<cv::xphoto::WhiteBalancer> _wb;
    Eigen::Vector3d _wb_gains;
    int _height = 0;
    int _rgb8_bytes = 0;
    int _width = 0;
    std::deque<Eigen::Vector3d> _wb_gain_history;

    void _reallocate_images() {
        if (_bayer_image.rows != _height || _bayer_image.cols != _width) {
            _bayer_image = cv::Mat(_height, _width, CV_16UC1);
            _rgb_image = cv::Mat(_height, _width, CV_16UC3);
            _rgb_image_8bit = cv::Mat(_height, _width, CV_8UC3);
            _rgb_image_out = cv::Mat(_height, _width, CV_8UC3);

            if (_use_cuda) {
                _gpu_bayer_image = cv::cuda::GpuMat(_height, _width, CV_16UC1);
                _gpu_rgb_image = cv::cuda::GpuMat(_height, _width, CV_16UC3);
                _gpu_rgb_image_8bit = cv::cuda::GpuMat(_height, _width, CV_8UC3);
            }
        }
    }

    void _apply_white_balance_cpu() {
        std::vector<cv::Mat> channels;
        cv::split(_rgb_image, channels);
        Eigen::Vector3d avg_gains(0.0, 0.0, 0.0);

        if (_use_grayworld_wb) {
            float means[3] = { 0.0f }, mean_gw = 0.0f;
            for (size_t i = 0; i < 3; ++i) {
                means[i] = cv::mean(channels[i])[0];
                mean_gw += means[i];
            }
            mean_gw /= 3.0f;

            Eigen::Vector3d wb_gains(mean_gw / means[0], mean_gw / means[1], mean_gw / means[2]);
            _add_gain_to_history(wb_gains);

            avg_gains = _compute_average_gains();

        } else {
            avg_gains = _wb_gains;
        }

        for (uint8_t i = 0; i < 3; ++i) {
            channels[i].convertTo(channels[i], CV_8U, avg_gains[i] / 255.0);
        }
        ROS_DEBUG("BGR gains: %.3f %.3f %.3f", avg_gains[0], avg_gains[1], avg_gains[2]);
        cv::merge(channels, _rgb_image_out);
    }

    void _apply_white_balance_gpu() {
        cv::cuda::split(_gpu_rgb_image, _gpu_rgb_channels);

        for (int i = 0; i < 3; ++i) {
            _gpu_rgb_channels[i].convertTo(_gpu_rgb_channels[i], CV_8U, _wb_gains[i] / 255.0);
        }

        cv::cuda::merge(_gpu_rgb_channels, _gpu_rgb_image_8bit);
    }

    void _add_gain_to_history(const Eigen::Vector3d& gain) {
        if (_wb_gain_history.size() >= _wb_N) {
            _wb_gain_history.pop_front();
        }
        _wb_gain_history.push_back(gain);
    }

    Eigen::Vector3d _compute_average_gains() const {
        Eigen::Vector3d sum_gains(0.0, 0.0, 0.0);
        for (const auto& gain : _wb_gain_history) {
            sum_gains += gain;
        }
        return sum_gains / static_cast<double>(_wb_gain_history.size());
    }

    std::vector<cv::cuda::GpuMat> _gpu_rgb_channels;
};

}  // namespace formats
}  // namespace usb_cam
