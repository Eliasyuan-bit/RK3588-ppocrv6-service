#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "ppocrv6_engine.h"

namespace {

void PrintUsage(const char* program) {
  std::cerr << "Usage: " << program
            << " --models <model-dir> --input <image> [--dict <dict.txt>]"
               " [--output <result.json>] [--disable-cls]\n";
}

std::string EscapeJson(const std::string& value) {
  std::string out;
  out.reserve(value.size() + 8);
  for (unsigned char ch : value) {
    switch (ch) {
      case '\\': out += "\\\\"; break;
      case '"': out += "\\\""; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (ch < 0x20) {
          char hex[7];
          std::snprintf(hex, sizeof(hex), "\\u%04x", ch);
          out += hex;
        } else {
          out += static_cast<char>(ch);
        }
    }
  }
  return out;
}

void WriteJson(std::ostream& out, const std::vector<ppocrv6::TextResult>& results) {
  out << "{\n  \"texts\": [\n";
  for (size_t i = 0; i < results.size(); ++i) {
    const auto& item = results[i];
    out << "    {\"text\": \"" << EscapeJson(item.text) << "\", \"rec_score\": "
        << item.rec_score << ", \"det_score\": " << item.det_score
        << ", \"rotated_180\": " << (item.rotated_180 ? "true" : "false")
        << ", \"box\": [";
    for (size_t point = 0; point < item.box.size(); ++point) {
      out << item.box[point] << (point + 1 == item.box.size() ? "" : ", ");
    }
    out << "]}" << (i + 1 == results.size() ? "\n" : ",\n");
  }
  out << "  ]\n}\n";
}

}  // namespace

int main(int argc, char** argv) {
  ppocrv6::EngineConfig config;
  std::string image_path;
  std::string output_path;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if ((arg == "--models" || arg == "--input" || arg == "--dict" || arg == "--output") && i + 1 < argc) {
      const std::string value = argv[++i];
      if (arg == "--models") config.model_dir = value;
      if (arg == "--input") image_path = value;
      if (arg == "--dict") config.dictionary_path = value;
      if (arg == "--output") output_path = value;
    } else if (arg == "--disable-cls") {
      config.enable_cls = false;
    } else {
      PrintUsage(argv[0]);
      return 2;
    }
  }
  if (config.model_dir.empty() || image_path.empty()) {
    PrintUsage(argv[0]);
    return 2;
  }
  if (config.dictionary_path.empty()) config.dictionary_path = config.model_dir + "/ppocrv6_dict.txt";
  try {
    ppocrv6::OcrEngine engine(config);
    engine.Initialize();
    const auto results = engine.RecognizeFile(image_path);
    if (output_path.empty()) {
      WriteJson(std::cout, results);
    } else {
      std::ofstream output(output_path);
      if (!output) throw std::runtime_error("cannot open output: " + output_path);
      WriteJson(output, results);
      std::cout << "wrote " << results.size() << " text boxes to " << output_path << '\n';
    }
  } catch (const std::exception& error) {
    std::cerr << "ppocrv6_ocr: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
