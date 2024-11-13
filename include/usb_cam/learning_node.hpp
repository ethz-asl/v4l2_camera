#ifndef USB_CAM_IMAGE_PROCESSING_NODE_HPP
#define USB_CAM_IMAGE_PROCESSING_NODE_HPP

#include <memory>
#include <string>
#include <vector>

#include "ros/ros.h"
#include "image_transport/image_transport.h"
#include "sensor_msgs/Image.h"
#include "usb_cam/learning/interface.hpp"
#include "usb_cam/learning/depth_anything_v2.hpp"

namespace usb_cam {

class ImageProcessingNode {
public:
    ImageProcessingNode();
    virtual ~ImageProcessingNode();
    bool spin();

private:
    image_transport::Subscriber _image_sub;
    ros::NodeHandle _node;
    std::string _dav2_file, _dav2_topic;
    std::vector<std::unique_ptr<LearningInterface>> _networks;
    void _image_callback(const sensor_msgs::ImageConstPtr& image_msg);
};

}  // namespace usb_cam

#endif  // USB_CAM_IMAGE_PROCESSING_NODE_HPP
