#include "ppocrv6_engine.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <rknn_api.h>

namespace ppocrv6 {
namespace {

[[noreturn]] void Fail(const std::string& message) { throw std::runtime_error(message); }

std::vector<unsigned char> ReadBinary(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) Fail("cannot open model: " + path);
  const auto length = file.tellg();
  if (length <= 0) Fail("empty model: " + path);
  std::vector<unsigned char> data(static_cast<size_t>(length));
  file.seekg(0);
  if (!file.read(reinterpret_cast<char*>(data.data()), length)) Fail("cannot read model: " + path);
  return data;
}

uint16_t FloatToHalf(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000U;
  const int exponent = static_cast<int>((bits >> 23) & 0xffU) - 127 + 15;
  uint32_t mantissa = bits & 0x7fffffU;
  if (exponent <= 0) {
    if (exponent < -10) return static_cast<uint16_t>(sign);
    mantissa = (mantissa | 0x800000U) >> (1 - exponent);
    return static_cast<uint16_t>(sign | ((mantissa + 0x1000U) >> 13));
  }
  if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00U);
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) |
                               ((mantissa + 0x1000U) >> 13));
}

class RknnContext {
 public:
  RknnContext() = default;
  ~RknnContext() { Reset(); }
  RknnContext(const RknnContext&) = delete;
  RknnContext& operator=(const RknnContext&) = delete;

  void Init(const std::string& path) {
    const auto model = ReadBinary(path);
    const int ret = rknn_init(&ctx_, const_cast<unsigned char*>(model.data()), model.size(), 0, nullptr);
    if (ret != RKNN_SUCC) Fail("rknn_init failed for " + path + ": " + std::to_string(ret));
    QueryMetadata(path);
  }

  void SetCoreMask(rknn_core_mask mask, const std::string& name) {
    const int ret = rknn_set_core_mask(ctx_, mask);
    if (ret != RKNN_SUCC) Fail("rknn_set_core_mask failed for " + name + ": " + std::to_string(ret));
  }

  std::vector<float> Run(const cv::Mat& bgr) const {
    if (bgr.empty() || bgr.type() != CV_8UC3 || !bgr.isContinuous()) Fail("RKNN input must be contiguous BGR uint8");
    if (bgr.cols != width_ || bgr.rows != height_) Fail("RKNN input dimension mismatch");

    cv::Mat input_float;
    bgr.convertTo(input_float, CV_32FC3);
    std::vector<uint16_t> normalized(input_float.total() * 3);
    const float* source = input_float.ptr<float>();
    for (size_t i = 0; i < normalized.size(); ++i) normalized[i] = FloatToHalf(source[i]);
    rknn_input input{};
    input.index = 0;
    input.type = RKNN_TENSOR_FLOAT16;
    input.fmt = RKNN_TENSOR_NHWC;
    input.size = static_cast<uint32_t>(normalized.size() * sizeof(uint16_t));
    input.buf = normalized.data();
    int ret = rknn_inputs_set(ctx_, 1, &input);
    if (ret != RKNN_SUCC) Fail("rknn_inputs_set failed: " + std::to_string(ret));
    ret = rknn_run(ctx_, nullptr);
    if (ret != RKNN_SUCC) Fail("rknn_run failed: " + std::to_string(ret));

    rknn_output output{};
    output.want_float = 1;
    ret = rknn_outputs_get(ctx_, 1, &output, nullptr);
    if (ret != RKNN_SUCC) Fail("rknn_outputs_get failed: " + std::to_string(ret));
    std::vector<float> result(output_.n_elems);
    std::copy_n(static_cast<const float*>(output.buf), output_.n_elems, result.data());
    rknn_outputs_release(ctx_, 1, &output);
    return result;
  }

  int width() const { return width_; }
  int height() const { return height_; }
  const rknn_tensor_attr& output() const { return output_; }

 private:
  void QueryMetadata(const std::string& name) {
    rknn_input_output_num io{};
    int ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io));
    if (ret != RKNN_SUCC || io.n_input != 1 || io.n_output != 1) Fail("unsupported I/O shape for " + name);
    input_.index = 0;
    output_.index = 0;
    ret = rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &input_, sizeof(input_));
    if (ret != RKNN_SUCC) Fail("cannot query input metadata for " + name);
    ret = rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &output_, sizeof(output_));
    if (ret != RKNN_SUCC) Fail("cannot query output metadata for " + name);
    if (input_.fmt != RKNN_TENSOR_NHWC || input_.n_dims != 4 || input_.dims[0] != 1 || input_.dims[3] != 3) {
      Fail("only static NHWC RGB models are supported: " + name);
    }
    height_ = input_.dims[1];
    width_ = input_.dims[2];
  }

  void Reset() {
    if (ctx_ != 0) rknn_destroy(ctx_);
    ctx_ = 0;
  }

  mutable rknn_context ctx_ = 0;
  rknn_tensor_attr input_{};
  rknn_tensor_attr output_{};
  int width_ = 0;
  int height_ = 0;
};

float Sigmoid(float value) {
  if (value >= 0.0F) return 1.0F / (1.0F + std::exp(-value));
  const float exp_value = std::exp(value);
  return exp_value / (1.0F + exp_value);
}

std::array<cv::Point2f, 4> OrderQuad(std::array<cv::Point2f, 4> points) {
  std::array<cv::Point2f, 4> ordered{};
  std::array<float, 4> sums{};
  std::array<float, 4> diffs{};
  for (int i = 0; i < 4; ++i) {
    sums[i] = points[i].x + points[i].y;
    diffs[i] = points[i].x - points[i].y;
  }
  ordered[0] = points[std::min_element(sums.begin(), sums.end()) - sums.begin()];
  ordered[2] = points[std::max_element(sums.begin(), sums.end()) - sums.begin()];
  ordered[1] = points[std::max_element(diffs.begin(), diffs.end()) - diffs.begin()];
  ordered[3] = points[std::min_element(diffs.begin(), diffs.end()) - diffs.begin()];
  return ordered;
}

cv::Mat ResizePadRgb(const cv::Mat& rgb, int height, int width, float* scale_out = nullptr) {
  const float scale = std::min(static_cast<float>(width) / rgb.cols, static_cast<float>(height) / rgb.rows);
  const int resized_width = std::max(1, static_cast<int>(std::round(rgb.cols * scale)));
  const int resized_height = std::max(1, static_cast<int>(std::round(rgb.rows * scale)));
  cv::Mat resized;
  cv::resize(rgb, resized, cv::Size(resized_width, resized_height));
  cv::Mat padded(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
  resized.copyTo(padded(cv::Rect(0, 0, resized_width, resized_height)));
  if (scale_out != nullptr) *scale_out = scale;
  return padded;
}

// PP-OCR recognition expects glyphs to use the full input height.  Unlike the
// Det/Cls letterbox path above, never shrink a long text line merely to make it
// fit the 320-pixel recognition width: resize to 48 pixels high first, then
// split the normalized strip into overlapping 320-pixel inputs.
std::vector<cv::Mat> ResizeForRec(const cv::Mat& bgr, int height, int width) {
  if (bgr.empty() || bgr.type() != CV_8UC3) Fail("Rec input must be BGR uint8");
  const float scale = static_cast<float>(height) / static_cast<float>(bgr.rows);
  const int resized_width = std::max(1, static_cast<int>(std::round(bgr.cols * scale)));
  cv::Mat resized;
  cv::resize(bgr, resized, cv::Size(resized_width, height));

  const auto make_input = [&](int left) {
    const int copy_width = std::min(width, resized.cols - left);
    cv::Mat input(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    resized(cv::Rect(left, 0, copy_width, height)).copyTo(input(cv::Rect(0, 0, copy_width, height)));
    return input;
  };

  if (resized_width <= width) return {make_input(0)};

  constexpr int kOverlapPixels = 32;
  const int stride = width - kOverlapPixels;
  std::vector<cv::Mat> inputs;
  for (int left = 0;;) {
    inputs.push_back(make_input(left));
    if (left + width >= resized_width) break;

    const int final_left = resized_width - width;
    int next_left = left + stride;
    if (next_left + width >= resized_width) next_left = final_left;
    // A final shift smaller than the overlap contributes almost no new glyph
    // pixels. The previous window already covers everything except that tail.
    if (next_left <= left || final_left - left < kOverlapPixels) break;
    left = next_left;
  }
  return inputs;
}

float FastBoxScore(const cv::Mat& probability, const std::array<cv::Point2f, 4>& box) {
  std::vector<cv::Point> contour;
  contour.reserve(4);
  for (const auto& point : box) contour.emplace_back(cvRound(point.x), cvRound(point.y));
  const cv::Rect rect = cv::boundingRect(contour) & cv::Rect(0, 0, probability.cols, probability.rows);
  if (rect.empty()) return 0.0F;
  cv::Mat mask = cv::Mat::zeros(rect.height, rect.width, CV_8UC1);
  for (auto& point : contour) point -= rect.tl();
  cv::fillConvexPoly(mask, contour, cv::Scalar(255));
  return static_cast<float>(cv::mean(probability(rect), mask)[0]);
}

struct DetBox {
  std::array<cv::Point2f, 4> points{};
  float score = 0.0F;
};

std::vector<DetBox> DecodeDet(const std::vector<float>& logits, int height, int width, const EngineConfig& config,
                              float scale, int source_width, int source_height) {
  if (logits.size() != static_cast<size_t>(height * width)) Fail("unexpected Det output shape");
  cv::Mat probability(height, width, CV_32FC1);
  cv::Mat bitmap(height, width, CV_8UC1);
  for (int i = 0; i < height * width; ++i) {
    const float probability_value = Sigmoid(logits[i]);
    probability.at<float>(i / width, i % width) = probability_value;
    bitmap.at<unsigned char>(i / width, i % width) = probability_value >= config.det_threshold ? 255 : 0;
  }
  cv::dilate(bitmap, bitmap, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2)));
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(bitmap, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
  std::vector<DetBox> decoded;
  decoded.reserve(std::min<size_t>(contours.size(), 1000));
  for (size_t i = 0; i < contours.size() && decoded.size() < 1000; ++i) {
    if (contours[i].size() < 3) continue;
    cv::RotatedRect rect = cv::minAreaRect(contours[i]);
    if (std::min(rect.size.width, rect.size.height) < 3.0F) continue;
    std::array<cv::Point2f, 4> points{};
    rect.points(points.data());
    points = OrderQuad(points);
    const float score = FastBoxScore(probability, points);
    if (score < config.det_box_threshold) continue;
    // A light-weight DB unclip approximation. It keeps the perspective crop stable
    // without depending on a polygon-offset implementation.
    rect.size.width *= config.det_unclip_ratio;
    rect.size.height *= config.det_unclip_ratio;
    rect.points(points.data());
    points = OrderQuad(points);
    for (auto& point : points) {
      point.x = std::clamp(point.x / scale, 0.0F, static_cast<float>(source_width - 1));
      point.y = std::clamp(point.y / scale, 0.0F, static_cast<float>(source_height - 1));
    }
    decoded.push_back({points, score});
  }
  return decoded;
}

std::vector<int> TileStarts(int extent, int tile_size, int overlap) {
  if (extent <= tile_size) return {0};
  const int stride = tile_size - overlap;
  if (stride <= 0) Fail("Det tile overlap must be smaller than the tile size");
  std::vector<int> starts;
  for (int start = 0;;) {
    starts.push_back(start);
    if (start + tile_size >= extent) break;
    const int final_start = extent - tile_size;
    int next_start = start + stride;
    if (next_start + tile_size >= extent) next_start = final_start;
    // Avoid a nearly identical penultimate/final tile. Near the right/bottom
    // edge it is preferable to reduce the requested overlap slightly instead
    // of running two windows that differ by only a few pixels.
    else if (final_start - next_start < overlap) next_start = final_start;
    if (next_start <= start) break;
    start = next_start;
  }
  return starts;
}

void OffsetDetBox(DetBox* box, int left, int top) {
  for (auto& point : box->points) {
    point.x += left;
    point.y += top;
  }
}

void ClampDetBox(DetBox* box, int source_width, int source_height) {
  for (auto& point : box->points) {
    point.x = std::clamp(point.x, 0.0F, static_cast<float>(source_width - 1));
    point.y = std::clamp(point.y, 0.0F, static_cast<float>(source_height - 1));
  }
  box->points = OrderQuad(box->points);
}

cv::Rect2f DetBounds(const DetBox& box) {
  float min_x = box.points[0].x;
  float max_x = min_x;
  float min_y = box.points[0].y;
  float max_y = min_y;
  for (const auto& point : box.points) {
    min_x = std::min(min_x, point.x);
    max_x = std::max(max_x, point.x);
    min_y = std::min(min_y, point.y);
    max_y = std::max(max_y, point.y);
  }
  return {min_x, min_y, std::max(0.0F, max_x - min_x), std::max(0.0F, max_y - min_y)};
}

float DetIoU(const DetBox& left, const DetBox& right) {
  const cv::Rect2f a = DetBounds(left);
  const cv::Rect2f b = DetBounds(right);
  const float inter_left = std::max(a.x, b.x);
  const float inter_top = std::max(a.y, b.y);
  const float inter_right = std::min(a.x + a.width, b.x + b.width);
  const float inter_bottom = std::min(a.y + a.height, b.y + b.height);
  const float inter_width = std::max(0.0F, inter_right - inter_left);
  const float inter_height = std::max(0.0F, inter_bottom - inter_top);
  const float intersection = inter_width * inter_height;
  const float union_area = a.width * a.height + b.width * b.height - intersection;
  return union_area > 0.0F ? intersection / union_area : 0.0F;
}

float DetIntersectionArea(const DetBox& left, const DetBox& right) {
  const cv::Rect2f a = DetBounds(left);
  const cv::Rect2f b = DetBounds(right);
  const float width = std::max(0.0F, std::min(a.x + a.width, b.x + b.width) - std::max(a.x, b.x));
  const float height = std::max(0.0F, std::min(a.y + a.height, b.y + b.height) - std::max(a.y, b.y));
  return width * height;
}

float DetContainment(const DetBox& left, const DetBox& right) {
  const cv::Rect2f a = DetBounds(left);
  const cv::Rect2f b = DetBounds(right);
  const float smaller_area = std::min(a.width * a.height, b.width * b.height);
  return smaller_area > 0.0F ? DetIntersectionArea(left, right) / smaller_area : 0.0F;
}

bool IsSameDetTextLine(const DetBox& left, const DetBox& right) {
  const cv::Rect2f a = DetBounds(left);
  const cv::Rect2f b = DetBounds(right);
  if (a.height <= 0.0F || b.height <= 0.0F || a.width <= 0.0F || b.width <= 0.0F) return false;
  const float vertical_overlap = std::max(0.0F, std::min(a.y + a.height, b.y + b.height) - std::max(a.y, b.y));
  if (vertical_overlap / std::min(a.height, b.height) < 0.70F) return false;
  const float horizontal_overlap =
      std::max(0.0F, std::min(a.x + a.width, b.x + b.width) - std::max(a.x, b.x));
  // Text boxes from adjacent columns normally have a gap. Requiring a material
  // horizontal overlap restricts stitching to partial boxes produced by two tiles.
  return horizontal_overlap / std::min(a.width, b.width) >= 0.15F;
}

DetBox MergeDetBoxes(const DetBox& left, const DetBox& right) {
  std::vector<cv::Point2f> points;
  points.reserve(8);
  points.insert(points.end(), left.points.begin(), left.points.end());
  points.insert(points.end(), right.points.begin(), right.points.end());
  const cv::RotatedRect rect = cv::minAreaRect(points);
  DetBox merged{};
  rect.points(merged.points.data());
  merged.points = OrderQuad(merged.points);
  merged.score = std::max(left.score, right.score);
  return merged;
}

std::vector<DetBox> DeduplicateDetBoxes(std::vector<DetBox> boxes, int source_width, int source_height) {
  std::sort(boxes.begin(), boxes.end(), [](const DetBox& left, const DetBox& right) {
    return left.score > right.score;
  });

  // A line crossing the overlap between two Det tiles can produce two partial,
  // overlapping boxes. Their IoU is often too low for ordinary NMS. Consolidate
  // them before recognition so Rec sees the complete line exactly once.
  bool changed = true;
  while (changed) {
    changed = false;
    std::vector<bool> consumed(boxes.size(), false);
    std::vector<DetBox> consolidated;
    consolidated.reserve(boxes.size());
    for (size_t i = 0; i < boxes.size(); ++i) {
      if (consumed[i]) continue;
      DetBox current = boxes[i];
      for (size_t j = i + 1; j < boxes.size(); ++j) {
        if (consumed[j]) continue;
        const bool same_line = IsSameDetTextLine(current, boxes[j]);
        const float containment = same_line ? DetContainment(current, boxes[j]) : 0.0F;
        if (DetIoU(current, boxes[j]) >= 0.50F || containment >= 0.80F) {
          const cv::Rect2f current_bounds = DetBounds(current);
          const cv::Rect2f candidate_bounds = DetBounds(boxes[j]);
          if (candidate_bounds.area() > current_bounds.area()) current = boxes[j];
          consumed[j] = true;
          changed = true;
        } else if (same_line) {
          current = MergeDetBoxes(current, boxes[j]);
          consumed[j] = true;
          changed = true;
        }
      }
      ClampDetBox(&current, source_width, source_height);
      consolidated.push_back(std::move(current));
    }
    boxes = std::move(consolidated);
  }
  return boxes;
}

std::vector<DetBox> DetectTextBoxes(const cv::Mat& bgr, const RknnContext& det, const EngineConfig& config) {
  if (config.det_tile_size <= 0 || config.det_tile_overlap < 0) Fail("invalid Det tile configuration");
  const auto x_starts = TileStarts(bgr.cols, config.det_tile_size, config.det_tile_overlap);
  const auto y_starts = TileStarts(bgr.rows, config.det_tile_size, config.det_tile_overlap);
  std::vector<DetBox> boxes;
  for (const int top : y_starts) {
    for (const int left : x_starts) {
      const int tile_width = std::min(config.det_tile_size, bgr.cols - left);
      const int tile_height = std::min(config.det_tile_size, bgr.rows - top);
      const cv::Mat tile = bgr(cv::Rect(left, top, tile_width, tile_height));
      float scale = 1.0F;
      const cv::Mat input = ResizePadRgb(tile, det.height(), det.width(), &scale);
      auto tile_boxes = DecodeDet(det.Run(input), det.height(), det.width(), config, scale, tile.cols, tile.rows);
      for (auto& box : tile_boxes) {
        OffsetDetBox(&box, left, top);
        boxes.push_back(std::move(box));
      }
    }
  }
  return DeduplicateDetBoxes(std::move(boxes), bgr.cols, bgr.rows);
}

cv::Mat PerspectiveCrop(const cv::Mat& rgb, std::array<cv::Point2f, 4> box) {
  for (auto& point : box) {
    point.x = std::clamp(point.x, 0.0F, static_cast<float>(rgb.cols - 1));
    point.y = std::clamp(point.y, 0.0F, static_cast<float>(rgb.rows - 1));
  }
  const float crop_width = std::max(1.0F, std::hypot(box[1].x - box[0].x, box[1].y - box[0].y));
  const float crop_height = std::max(1.0F, std::hypot(box[3].x - box[0].x, box[3].y - box[0].y));
  const std::array<cv::Point2f, 4> target = {
      cv::Point2f(0, 0), cv::Point2f(crop_width - 1, 0),
      cv::Point2f(crop_width - 1, crop_height - 1), cv::Point2f(0, crop_height - 1)};
  cv::Mat transform = cv::getPerspectiveTransform(box.data(), target.data());
  cv::Mat crop;
  cv::warpPerspective(rgb, crop, transform, cv::Size(cvRound(crop_width), cvRound(crop_height)),
                      cv::INTER_LINEAR, cv::BORDER_REPLICATE);
  if (crop.rows >= crop.cols * 1.5F) {
    cv::transpose(crop, crop);
    cv::flip(crop, crop, 0);
  }
  return crop;
}

std::vector<std::string> ReadDictionary(const std::string& path) {
  std::ifstream file(path);
  if (!file) Fail("cannot open dictionary: " + path);
  std::vector<std::string> dictionary;
  std::string line;
  while (std::getline(file, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    dictionary.push_back(line);
  }
  if (dictionary.empty()) Fail("empty dictionary: " + path);
  // PP-OCR's use_space_char appends this token at runtime; it is not present
  // in the shipped 18,708-line basic dictionary.  The model has CTC blank +
  // 18,709 characters = 18,710 output channels.
  if (std::find(dictionary.begin(), dictionary.end(), " ") == dictionary.end()) dictionary.push_back(" ");
  return dictionary;
}

float SoftmaxProbability(const float* logits, int length, int selected) {
  const float max_value = *std::max_element(logits, logits + length);
  float sum = 0.0F;
  for (int i = 0; i < length; ++i) sum += std::exp(logits[i] - max_value);
  return std::exp(logits[selected] - max_value) / sum;
}

std::pair<std::string, float> DecodeRec(const std::vector<float>& logits, const rknn_tensor_attr& output,
                                        const std::vector<std::string>& dictionary) {
  const int channels = output.dims[output.n_dims - 1];
  if (channels != static_cast<int>(dictionary.size()) + 1 || logits.size() % channels != 0) {
    Fail("Rec output channel count does not match ppocrv6_dict.txt: channels=" + std::to_string(channels) +
         ", dictionary=" + std::to_string(dictionary.size()) + ", elements=" + std::to_string(logits.size()));
  }
  const int time_steps = static_cast<int>(logits.size()) / channels;
  std::string text;
  float confidence_sum = 0.0F;
  int confidence_count = 0;
  int previous_index = 0;
  for (int time = 0; time < time_steps; ++time) {
    const float* row = logits.data() + time * channels;
    const int index = static_cast<int>(std::max_element(row, row + channels) - row);
    if (index != 0 && index != previous_index) {
      text += dictionary[index - 1];
      confidence_sum += SoftmaxProbability(row, channels, index);
      ++confidence_count;
    }
    previous_index = index;
  }
  return {text, confidence_count == 0 ? 0.0F : confidence_sum / confidence_count};
}

std::vector<std::string> SplitUtf8(const std::string& text) {
  std::vector<std::string> characters;
  for (size_t offset = 0; offset < text.size();) {
    const unsigned char byte = static_cast<unsigned char>(text[offset]);
    size_t length = 1;
    if ((byte & 0xE0U) == 0xC0U) length = 2;
    else if ((byte & 0xF0U) == 0xE0U) length = 3;
    else if ((byte & 0xF8U) == 0xF0U) length = 4;
    if (offset + length > text.size()) length = 1;
    characters.push_back(text.substr(offset, length));
    offset += length;
  }
  return characters;
}

std::string MergeRecText(const std::string& left, const std::string& right) {
  if (left.empty()) return right;
  if (right.empty()) return left;
  const auto left_chars = SplitUtf8(left);
  const auto right_chars = SplitUtf8(right);
  const size_t max_overlap = std::min(left_chars.size(), right_chars.size());
  size_t overlap = 0;
  for (size_t length = max_overlap; length > 0; --length) {
    if (std::equal(left_chars.end() - length, left_chars.end(), right_chars.begin())) {
      overlap = length;
      break;
    }
  }
  std::string merged = left;
  for (size_t index = overlap; index < right_chars.size(); ++index) merged += right_chars[index];
  return merged;
}

std::pair<std::string, float> DecodeRecSegments(const std::vector<cv::Mat>& inputs, const RknnContext& rec,
                                                 const std::vector<std::string>& dictionary) {
  std::string text;
  float confidence_sum = 0.0F;
  int confidence_count = 0;
  for (const auto& input : inputs) {
    const auto decoded = DecodeRec(rec.Run(input), rec.output(), dictionary);
    if (decoded.first.empty()) continue;
    text = MergeRecText(text, decoded.first);
    confidence_sum += decoded.second;
    ++confidence_count;
  }
  return {text, confidence_count == 0 ? 0.0F : confidence_sum / confidence_count};
}

bool ShouldRotate180(const std::vector<float>& logits) {
  if (logits.size() != 2) Fail("unexpected Cls output shape");
  const int index = logits[1] > logits[0] ? 1 : 0;
  return index == 1 && SoftmaxProbability(logits.data(), 2, index) > 0.90F;
}

bool ReadingOrder(const DetBox& left, const DetBox& right) {
  const float left_y = (left.points[0].y + left.points[1].y) * 0.5F;
  const float right_y = (right.points[0].y + right.points[1].y) * 0.5F;
  if (std::abs(left_y - right_y) > 10.0F) return left_y < right_y;
  return left.points[0].x < right.points[0].x;
}

std::array<int, 8> ToResultBox(const std::array<cv::Point2f, 4>& points) {
  std::array<int, 8> box{};
  for (size_t i = 0; i < points.size(); ++i) {
    box[i * 2] = cvRound(points[i].x);
    box[i * 2 + 1] = cvRound(points[i].y);
  }
  return box;
}

}  // namespace

class OcrEngine::Impl {
 public:
  explicit Impl(EngineConfig input_config) : config(std::move(input_config)) {}

  void Initialize() {
    if (initialized) return;
    if (!std::filesystem::is_directory(config.model_dir)) Fail("model_dir is not a directory: " + config.model_dir);
    dictionary = ReadDictionary(config.dictionary_path);

    det_all.Init(config.model_dir + "/ppocrv6_det_480x480_logits_fp.rknn");
    det_all.SetCoreMask(RKNN_NPU_CORE_0_1_2, "Det");

    cls.Init(config.model_dir + "/ppocrv6_cls_48x192_logits_fp.rknn");
    cls.SetCoreMask(RKNN_NPU_CORE_0, "Cls");

    rec.Init(config.model_dir + "/ppocrv6_rec_48x320_logits_fp.rknn");
    rec.SetCoreMask(RKNN_NPU_CORE_0, "Rec");
    initialized = true;
  }

  std::vector<TextResult> RecognizeFile(const std::string& image_path) {
    if (!initialized) Fail("engine is not initialized");
    cv::Mat bgr = cv::imread(image_path, cv::IMREAD_COLOR);
    if (bgr.empty()) Fail("cannot read image: " + image_path);
    // The RKNN conversion used quant_img_RGB2BGR=False. Keep OpenCV's BGR
    // channel order instead of swapping it before the model input.
    const cv::Mat& rgb = bgr;

    std::vector<DetBox> boxes = DetectTextBoxes(rgb, det_all, config);
    std::sort(boxes.begin(), boxes.end(), ReadingOrder);

    struct CropTask { DetBox box; cv::Mat crop; bool rotated = false; };
    std::vector<CropTask> tasks;
    tasks.reserve(boxes.size());
    for (const auto& box : boxes) {
      cv::Mat crop = PerspectiveCrop(rgb, box.points);
      if (crop.empty()) continue;
      bool rotated = false;
      if (config.enable_cls) {
        const cv::Mat cls_input = ResizePadRgb(crop, cls.height(), cls.width());
        rotated = ShouldRotate180(cls.Run(cls_input));
        if (rotated) cv::rotate(crop, crop, cv::ROTATE_180);
      }
      tasks.push_back({box, crop, rotated});
    }

    std::vector<TextResult> results;
    results.reserve(tasks.size());
    for (const auto& task : tasks) {
      const auto rec_inputs = ResizeForRec(task.crop, rec.height(), rec.width());
      const auto decoded = DecodeRecSegments(rec_inputs, rec, dictionary);
      if (decoded.first.empty() || decoded.second < config.rec_score_threshold) continue;
      results.push_back({ToResultBox(task.box.points), task.box.score, decoded.second, task.rotated, decoded.first});
    }
    return results;
  }

  EngineConfig config;
  std::vector<std::string> dictionary;
  RknnContext det_all;
  RknnContext cls;
  RknnContext rec;
  bool initialized = false;
};

OcrEngine::OcrEngine(EngineConfig config) : impl_(std::make_unique<Impl>(std::move(config))) {}
OcrEngine::~OcrEngine() = default;
void OcrEngine::Initialize() { impl_->Initialize(); }
std::vector<TextResult> OcrEngine::RecognizeFile(const std::string& image_path) { return impl_->RecognizeFile(image_path); }

}  // namespace ppocrv6
