#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ppocrv6_engine.h"

namespace {

void PrintUsage(const char* program) {
  std::cerr << "Usage: " << program << " --models <model-dir> [--dict <dict.txt>] [--disable-cls]\n"
            << "Reads JSON Lines from stdin: {\"id\":\"request-1\",\"input\":\"/path/page.jpg\"}\n";
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
      default: out += static_cast<char>(ch); break;
    }
  }
  return out;
}

// Minimal JSON string reader for the fixed JSONL protocol. It handles escaped
// quotes and backslashes; request fields other than id/input are ignored.
std::string JsonStringField(const std::string& line, const std::string& name) {
  const std::string key = "\"" + name + "\"";
  size_t pos = line.find(key);
  if (pos == std::string::npos) return "";
  pos = line.find(':', pos + key.size());
  if (pos == std::string::npos) throw std::runtime_error("invalid JSON field: " + name);
  pos = line.find('"', pos + 1);
  if (pos == std::string::npos) throw std::runtime_error("field is not a JSON string: " + name);
  std::string value;
  bool escaped = false;
  for (++pos; pos < line.size(); ++pos) {
    const char ch = line[pos];
    if (escaped) {
      switch (ch) {
        case 'n': value += '\n'; break;
        case 'r': value += '\r'; break;
        case 't': value += '\t'; break;
        default: value += ch; break;
      }
      escaped = false;
    } else if (ch == '\\') {
      escaped = true;
    } else if (ch == '"') {
      return value;
    } else {
      value += ch;
    }
  }
  throw std::runtime_error("unterminated JSON string: " + name);
}

void WriteResponse(const std::string& id, const std::vector<ppocrv6::TextResult>& results) {
  std::cout << "{\"id\":\"" << EscapeJson(id) << "\",\"ok\":true,\"texts\":[";
  for (size_t i = 0; i < results.size(); ++i) {
    const auto& item = results[i];
    std::cout << (i == 0 ? "" : ",") << "{\"text\":\"" << EscapeJson(item.text)
              << "\",\"rec_score\":" << item.rec_score << ",\"det_score\":" << item.det_score
              << ",\"rotated_180\":" << (item.rotated_180 ? "true" : "false") << ",\"box\":[";
    for (size_t point = 0; point < item.box.size(); ++point) {
      std::cout << item.box[point] << (point + 1 == item.box.size() ? "" : ",");
    }
    std::cout << "]}";
  }
  std::cout << "]}" << std::endl;
}

void WriteError(const std::string& id, const std::string& error) {
  std::cout << "{\"id\":\"" << EscapeJson(id) << "\",\"ok\":false,\"error\":\""
            << EscapeJson(error) << "\"}" << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  ppocrv6::EngineConfig config;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if ((arg == "--models" || arg == "--dict") && i + 1 < argc) {
      const std::string value = argv[++i];
      if (arg == "--models") config.model_dir = value;
      if (arg == "--dict") config.dictionary_path = value;
    } else if (arg == "--disable-cls") {
      config.enable_cls = false;
    } else {
      PrintUsage(argv[0]);
      return 2;
    }
  }
  if (config.model_dir.empty()) {
    PrintUsage(argv[0]);
    return 2;
  }
  if (config.dictionary_path.empty()) config.dictionary_path = config.model_dir + "/ppocrv6_dict.txt";

  try {
    ppocrv6::OcrEngine engine(config);
    engine.Initialize();
    std::string request;
    while (std::getline(std::cin, request)) {
      if (request.empty()) continue;
      std::string id;
      try {
        id = JsonStringField(request, "id");
        const std::string input = JsonStringField(request, "input");
        if (input.empty()) throw std::runtime_error("missing required string field: input");
        WriteResponse(id, engine.RecognizeFile(input));
      } catch (const std::exception& error) {
        WriteError(id, error.what());
      }
    }
  } catch (const std::exception& error) {
    std::cerr << "ppocrv6_ocr_daemon: initialization failed: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
