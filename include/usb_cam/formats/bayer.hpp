#ifndef USB_CAM__FORMATS__BAYER_GRBG10_HPP_
#define USB_CAM__FORMATS__BAYER_GRBG10_HPP_

#include "linux/videodev2.h"
#include "usb_cam/formats/pixel_format_base.hpp"
#include "usb_cam/formats/utils.hpp"

namespace usb_cam
{
namespace formats
{

class BAYER_GRBG10 : public pixel_format_base
{
public:
  explicit BAYER_GRBG10(const format_arguments_t & args = format_arguments_t())
  : pixel_format_base(
      "bayer_grbg10",               // Format name
      V4L2_PIX_FMT_SGRBG10,         // V4L2 identifier for GRBG 10-bit Bayer
      usb_cam::constants::RGB8, // Converts to RGB8
      3,
      8,
      true),                        // True indicates that this needs a conversion to RGB8
    m_number_of_pixels(args.pixels)
  {}

  /// @brief Convert a BAYER_GRBG10 image to RGB8
  /// @param src pointer to source BAYER_GRBG10 image
  /// @param dest pointer to destination RGB8 image
  /// @param bytes_used number of bytes used by source image
  void convert(const char * & src, char * & dest, const int & bytes_used) override
  {
    (void)bytes_used;  // Not used in this case

    // Iterate over the image, performing the Bayer to RGB conversion.
    int i = 0;
    int j = 0;
    while (i < (m_number_of_pixels * 5 / 4)) {  // Each pixel is represented by 10 bits (5 bytes per 4 pixels)
      uint16_t pixel0 = (src[i] | (src[i + 1] << 8)) & 0x03FF;   // First 10-bit pixel
      uint16_t pixel1 = (src[i + 1] >> 2 | (src[i + 2] << 6)) & 0x03FF; // Second 10-bit pixel
      uint16_t pixel2 = (src[i + 2] >> 4 | (src[i + 3] << 4)) & 0x03FF; // Third 10-bit pixel
      uint16_t pixel3 = (src[i + 3] >> 6 | (src[i + 4] << 2)) & 0x03FF; // Fourth 10-bit pixel
      i += 5;

      // Now, perform Bayer interpolation for GRBG pattern to get RGB values.
      // This part is highly simplified and you'd normally use more sophisticated interpolation.
      dest[j++] = static_cast<unsigned char>((pixel0 >> 2) & 0xFF);  // Red
      dest[j++] = static_cast<unsigned char>((pixel1 >> 2) & 0xFF);  // Green
      dest[j++] = static_cast<unsigned char>((pixel2 >> 2) & 0xFF);  // Blue

      // Continue for remaining pixels (you'll need a proper interpolation algorithm for a real-world case)
      // Here it's a simple approach, but you could implement bilinear interpolation for more accurate conversion
    }
  }

private:
  int m_number_of_pixels;  // Number of pixels in the image
};

}  // namespace formats
}  // namespace usb_cam

#endif  // USB_CAM__FORMATS__BAYER_GRBG10_HPP_
