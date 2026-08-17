#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace ppocrv6 {

struct EngineConfig {
  std::string model_dir;
  std::string dictionary_path;
  bool enable_cls = true;
  float det_threshold = 0.30F;
  float det_box_threshold = 0.50F;
  float det_unclip_ratio = 1.60F;
  float rec_score_threshold = 0.50F;
};

struct TextResult {
  std::array<int, 8> box{};  // left-top, right-top, right-bottom, left-bottom
  float det_score = 0.0F;
  float rec_score = 0.0F;
  bool rotated_180 = false;
  std::string text;
};

class OcrEngine {
 public:
  explicit OcrEngine(EngineConfig config);
  ~OcrEngine();
  OcrEngine(const OcrEngine&) = delete;
  OcrEngine& operator=(const OcrEngine&) = delete;

  // Loads Det, Cls and Rec once. Det is bound to all three NPU cores;
  // Cls and Rec stay on Core 0 before accepting inference requests.
  void Initialize();
  std::vector<TextResult> RecognizeFile(const std::string& image_path);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace ppocrv6
