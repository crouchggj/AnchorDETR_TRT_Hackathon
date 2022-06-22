//
// Created by corey on 2022/4/2.
//

#ifndef CORNETNET_LITE_TRT_COMMON_H
#define CORNETNET_LITE_TRT_COMMON_H

#include <chrono>
#include <fstream>
#include <memory>
#include <sstream>
#include <vector>

namespace hackathon {
struct CommandParam {
  std::string model_path;
  bool use_half;
  bool use_int8;
};

struct InferDeleter {
  template <typename T>
  void operator()(T* obj) const {
    if (obj) {
      obj->destroy();
    }
  }
};

template <typename T>
using UniquePtr = std::unique_ptr<T, InferDeleter>;

// Logger for TensorRT info/warning/errors
class Logger : public nvinfer1::ILogger {
 public:
  Logger() : Logger(Severity::kINFO) {}

  Logger(Severity severity) : reportableSeverity(severity) {}

  nvinfer1::ILogger& getTRTLogger() noexcept { return *this; }

  void log(Severity severity, const char* msg) noexcept override {
    // suppress messages with severity enum value greater than the reportable
    if (severity > reportableSeverity) return;

    switch (severity) {
      case Severity::kINTERNAL_ERROR:
        std::cerr << "INTERNAL_ERROR: ";
        break;
      case Severity::kERROR:
        std::cerr << "ERROR: ";
        break;
      case Severity::kWARNING:
        std::cerr << "WARNING: ";
        break;
      case Severity::kINFO:
        std::cerr << "INFO: ";
        break;
      default:
        std::cerr << "UNKNOWN: ";
        break;
    }
    std::cerr << msg << std::endl;
  }

  Severity reportableSeverity{Severity::kWARNING};
};

/*
 * @brief: 判断文件是否存在
 * @param: [in] file : 文件路径
 * @ret:   true: 存在　false: 不存在
 * */
inline bool check_file_exist(const std::string& file) {
  std::ifstream f(file.c_str());
  return f.good();
}
}  // namespace hackathon

constexpr long double operator"" _GiB(long double val) {
  return val * (1 << 30);
}
constexpr long double operator"" _MiB(long double val) {
  return val * (1 << 20);
}
constexpr long double operator"" _KiB(long double val) {
  return val * (1 << 10);
}

// These is necessary if we want to be able to write 1_GiB instead of 1.0_GiB.
// Since the return type is signed, -1_GiB will work as expected.
constexpr long long int operator"" _GiB(unsigned long long val) {
  return val * (1 << 30);
}
constexpr long long int operator"" _MiB(unsigned long long val) {
  return val * (1 << 20);
}
constexpr long long int operator"" _KiB(unsigned long long val) {
  return val * (1 << 10);
}

#endif  // COMMON_H
