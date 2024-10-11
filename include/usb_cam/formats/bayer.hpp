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
        true),  // True indicates that this needs a conversion to RGB8
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

        _wb = cv::xphoto::createGrayworldWB();
        _use_grayworld_wb = (_wb_blue_gain < 1e-6f || _wb_green_gain < 1e-6f || _wb_red_gain < 1e-6f);
    }

    /// @brief Convert a BAYER_GRBG10 image to RGB8 with WB
    /// @param src pointer to source BAYER_GRBG10 image
    /// @param dest pointer to destination RGB8 image
    /// @param bytes_used number of bytes used by source image
    void convert(const char* &src, char* &dest, const int& bytes_used) override {
        (void)bytes_used;
        _reallocate_images();
        _bayer_image.data = (uchar*)src;

        // Demosaic and convert to 8-bit with white balance scaling
        if (_use_cuda && !_use_grayworld_wb) {
            _gpu_bayer_image.upload(_bayer_image);
            cv::cuda::cvtColor(_gpu_bayer_image, _gpu_rgb_image, cv::COLOR_BayerGR2BGR);
            _apply_white_balance_gpu();
            _gpu_rgb_image_8bit.download(_rgb_image_out);

        } else {
            cv::cvtColor(_bayer_image, _rgb_image, cv::COLOR_BayerGR2BGR);
            _apply_white_balance_cpu();
        }

        // Copy the result to the destination buffer
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
    float _wb_blue_gain;
    float _wb_green_gain;
    float _wb_red_gain;
    int _height = 0;
    int _rgb8_bytes = 0;
    int _width = 0;
    std::deque<float> _wb_blue_gains;
    std::deque<float> _wb_green_gains;
    std::deque<float> _wb_red_gains;

    // Allocate/Reallocate images if needed
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

    // Apply white balance and 16-bit to 8-bit conversion on CPU
    void _apply_white_balance_cpu() {
        std::vector<cv::Mat> channels;
        cv::split(_rgb_image, channels);
        float avg_blue_gain = 0.0f, avg_green_gain = 0.0f, avg_red_gain = 0.0f;

        if (_use_grayworld_wb) {
            // Compute gray world white balance
            float means[3] = { 0.0f }, mean_gw = 0.0f;
            for (size_t i = 0; i < 3; ++i) {
                means[i] = cv::mean(channels[i])[0];
                mean_gw += means[i];
            }
            mean_gw /= 3.0f;

            // Compute gains
            float wb_blue_gain = mean_gw / means[0];
            float wb_green_gain = mean_gw / means[1];
            float wb_red_gain = mean_gw / means[2];

            // Maintain average gains using deque
            _add_gain_to_deque(_wb_blue_gains, wb_blue_gain);
            _add_gain_to_deque(_wb_green_gains, wb_green_gain);
            _add_gain_to_deque(_wb_red_gains, wb_red_gain);

            avg_blue_gain = _compute_average(_wb_blue_gains);
            avg_green_gain = _compute_average(_wb_green_gains);
            avg_red_gain = _compute_average(_wb_red_gains);

        } else {
            avg_blue_gain = _wb_blue_gain;
            avg_green_gain = _wb_green_gain;
            avg_red_gain = _wb_red_gain;
        }

        // Apply gains using convertTo
        for (int i = 0; i < 3; ++i) {
            float gain = (i == 0) ? avg_blue_gain : (i == 1) ? avg_green_gain : avg_red_gain;
            channels[i].convertTo(channels[i], CV_8U, gain / 256.0);  // 16-bit to 8-bit and apply gain
        }
        std::cout << "BGR gains: " << avg_blue_gain << ", " << avg_green_gain << ", "<< avg_red_gain << std::endl;

        cv::merge(channels, _rgb_image_out);
    }

    // Apply white balance and 16-bit to 8-bit conversion on GPU
    void _apply_white_balance_gpu() {
        cv::cuda::split(_gpu_rgb_image, _gpu_rgb_channels);

        // Upload custom white balance gains to GPU (manual scaling)
        _gpu_rgb_channels[0].convertTo(_gpu_rgb_channels[0], CV_8U, _wb_blue_gain / 256.0);
        _gpu_rgb_channels[1].convertTo(_gpu_rgb_channels[1], CV_8U, _wb_green_gain / 256.0);
        _gpu_rgb_channels[2].convertTo(_gpu_rgb_channels[2], CV_8U, _wb_red_gain / 256.0);

        cv::cuda::merge(_gpu_rgb_channels, _gpu_rgb_image_8bit);
    }

    // Add gain to deque and maintain window size N
    void _add_gain_to_deque(std::deque<float>& deque, float gain) {
        if (deque.size() >= _wb_N) deque.pop_front();
        deque.push_back(gain);
    }

    // Compute average gain from deque
    float _compute_average(const std::deque<float>& gains) {
        float sum = 0.0f;
        for (float gain : gains) {
            sum += gain;
        }
        return sum / gains.size();
    }

    // GPU channel storage
    std::vector<cv::cuda::GpuMat> _gpu_rgb_channels;
};

}  // namespace formats
}  // namespace usb_cam
