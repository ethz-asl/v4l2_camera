#include "usb_cam/learning_node.hpp"

namespace usb_cam {

ImageProcessingNode::ImageProcessingNode() : _node("~") {
    image_transport::ImageTransport it(_node);

    std::string input_image_topic;
    _node.param("input_image_topic", input_image_topic, std::string(""));
    if (!input_image_topic.empty()) {
        _image_sub = it.subscribe(input_image_topic, 1, &ImageProcessingNode::_image_callback, this);
    }

    _node.param("dav2_file", _dav2_file, std::string(""));
    _node.param("dav2_topic", _dav2_topic, std::string(""));

    // Initialize network if parameters are provided
    if (!_dav2_file.empty() && !_dav2_topic.empty()) {
        _networks.push_back(std::make_unique<DepthAnythingV2>(&_node, _dav2_file, _dav2_topic));
    }
}

ImageProcessingNode::~ImageProcessingNode() { }

void ImageProcessingNode::_image_callback(const sensor_msgs::ImageConstPtr& image_msg) {
    for (const auto& net : _networks) {
        net->set_input(*image_msg);
        net->predict();
        net->publish();
    }
}

bool ImageProcessingNode::spin() {
    ros::spin();
    return true;
}

}  // namespace usb_cam

int main(int argc, char** argv) {
    ros::init(argc, argv, "image_processing_node");
    usb_cam::ImageProcessingNode node;
    node.spin();
    return EXIT_SUCCESS;
}
