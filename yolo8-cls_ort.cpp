#include "yolo8-cls_ort.h"

using namespace std;
using namespace Ort;
using namespace lodepng;

typedef unsigned char uchar;
typedef unsigned int uint;

template <typename T>
std::vector<T> bilinear_interpolate(const std::vector<T>& image, int width, int height, int channels, double x, double y) {
  int x0 = floor(x);
  int y0 = floor(y);
  int x1 = x0 + 1;
  int y1 = y0 + 1;

  // Clamp coordinates to image boundaries
  x0 = max(0, min(x0, width - 1));
  y0 = max(0, min(y0, height - 1));
  x1 = max(0, min(x1, width - 1));
  y1 = max(0, min(y1, height - 1));

  int index00 = y0 * width * channels + x0 * channels;
  int index10 = y0 * width * channels + x1 * channels;
  int index01 = y1 * width * channels + x0 * channels;
  int index11 = y1 * width * channels + x1 * channels;

  double a = x - x0;
  double b = y - y0;

  std::vector<T> interpolated_pixel(channels);
  for (int channel = 0; channel < channels; ++channel) {
    T f00 = image[index00 + channel];
    T f10 = image[index10 + channel];
    T f01 = image[index01 + channel];
    T f11 = image[index11 + channel];

    interpolated_pixel[channel] = (1 - a) * (1 - b) * f00 + a * (1 - b) * f10 +
      (1 - a) * b * f01 + a * b * f11;
  }

  return interpolated_pixel;
}

template <typename T>
void resize_image(const vector<T>& image, vector<T>& resized_image, int width, int height, int channels, int new_width, int new_height) {

  double x_ratio = (double)(width - 1) / (new_width - 1);
  double y_ratio = (double)(height - 1) / (new_height - 1);

  for (int i = 0; i < new_height; ++i) {
    for (int j = 0; j < new_width; ++j) {
      double x = j * x_ratio;
      double y = i * y_ratio;

      std::vector<T> interpolated_pixel = bilinear_interpolate(image, width, height, channels, x, y);

      for (int channel = 0; channel < channels; ++channel) {
        resized_image[(i * new_width + j) * channels + channel] = interpolated_pixel[channel];
      }
    }
  }
}

int main(int argc, char* argv[])
{
  string img_path = argv[2];
  
#ifdef _WIN32  
  std::string str = argv[1];
  std::wstring wide_string = std::wstring(str.begin(), str.end());
  std::basic_string<ORTCHAR_T> model_file = std::basic_string<ORTCHAR_T>(wide_string);
#else
  std::string model_file = argv[1];
#endif

  Env env(ORT_LOGGING_LEVEL_WARNING, "test");
  SessionOptions session_options;
  AllocatorWithDefaultOptions allocator;

  //	ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));

  Session session( env, model_file.c_str(), session_options );

  size_t num_input_nodes = session.GetInputCount();
  vector<const char*> input_node_names;
  vector<AllocatedStringPtr> input_names_ptr;
  vector<const char*> output_node_names{"output0"};

  input_names_ptr.reserve(num_input_nodes);
  input_node_names.reserve(num_input_nodes);
  
  for(size_t i = 0 ; i < num_input_nodes ; i ++) {
    auto input_name = session.GetInputNameAllocated(i, allocator);
    input_node_names.push_back(input_name.get());
    input_names_ptr.push_back(std::move(input_name));
  }

  auto shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

  uint width;
  uint height;
  vector<uchar> img_data;
  vector<uchar> resized_data(shape[2] * shape[3] * 4);

  uint error = decode(img_data, width, height, img_path.c_str());

  if (error) {
    cerr << "Error while reading the image" << endl;
    return -1;
  }

  // Resize original image to (shape[2], shape[3])
  resize_image(img_data, resized_data, width, height, 4, shape[2], shape[3]);

  // Convert 4-channel image to 3-channel
  for (int i = resized_data.size() - 1; i >= 2; i -= 4)
    resized_data.erase(resized_data.begin() + i);

  float* rawdata = new float[shape[1] * shape[2] * shape[3] + 1];

  // Transpose from (w, h, c) to (c, w, h)
  for (int c = 0; c < shape[1]; c++)
    for (int w = 0; w < shape[2]; w++)
      for (int h = 0; h < shape[3]; h++)
        rawdata[c * shape[2] * shape[3] + w * shape[2] + h] = (float)resized_data[shape[1] * (shape[2] * h + w) + c] / 255.0;

  MemoryInfo memory_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  Value input_tensor = Value::CreateTensor<float>(memory_info, (float*)rawdata, shape[1] * shape[2] * shape[3], shape.data(), shape.size());
  vector<Value> output_tensors;

  RunOptions options = Ort::RunOptions{ nullptr };
  output_tensors = session.Run(options, input_node_names.data(), &input_tensor, num_input_nodes, output_node_names.data(), num_input_nodes);
  
  assert(output_tensors.size() == num_input_nodes && output_tensors.front().IsTensor());

  float* floatarr = output_tensors.front().GetTensorMutableData<float>();

  std::cout << floatarr[0] << " " << floatarr[1] << endl;
  return 0;
}
