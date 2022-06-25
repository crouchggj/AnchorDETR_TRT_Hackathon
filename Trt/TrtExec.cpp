//
// Created by corey on 2022/4/2.
//

#include <NvInferPlugin.h>
#include <NvOnnxParser.h>

#include <iostream>

#include "Logger.h"
#include "Utils.h"
#include "common.h"

using namespace nvinfer1;
using namespace std;

simplelogger::Logger* logger =
    simplelogger::LoggerFactory::CreateConsoleLogger(simplelogger::TRACE);

namespace {
hackathon::Logger gLogger;
constexpr const char* kTrtPlanName = "AnchorDETR.plan";
}  // namespace

bool OnnxToTrt(const hackathon::CommandParam& command_param) {
  auto builder = hackathon::UniquePtr<IBuilder>(createInferBuilder(gLogger));
  if (!builder) {
    std::cout << "create infer builder failure\n";
    return false;
  }

  const auto explicitBatch =
      1U << static_cast<uint32_t>(
          nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  auto network = hackathon::UniquePtr<INetworkDefinition>(
      builder->createNetworkV2(explicitBatch));
  if (!network) {
    std::cout << "create networkV2 failure\n";
    return false;
  }
  auto config =
      hackathon::UniquePtr<IBuilderConfig>(builder->createBuilderConfig());

  auto parser = hackathon::UniquePtr<nvonnxparser::IParser>(
      nvonnxparser::createParser(*network, gLogger));
  auto parsed =
      parser->parseFromFile(command_param.model_path.c_str(),
                            static_cast<int>(gLogger.reportableSeverity));
  if (!parsed) {
    cout << "parse model failure\n";
    return false;
  }

  config->setMaxWorkspaceSize(6_GiB);

  BuildEngineParam param;
  param.nChannel = network->getInput(0)->getDimensions().d[1];
  param.nHeight = network->getInput(0)->getDimensions().d[2];
  param.nWidth = network->getInput(0)->getDimensions().d[3];
  auto calib = make_unique<Calibrator>(1, &param, "int8_trt.cache");

  string prefix;
  if (command_param.use_half) {
    cout << "Use HALF Mode\n";
    config->setFlag(BuilderFlag::kFP16);
    config->setFlag(BuilderFlag::kSTRICT_TYPES);
    prefix = "HALF";
  } else if (command_param.use_int8) {
    cout << "Use INT8 Mode\n";
    config->setFlag(BuilderFlag::kINT8);
    config->setInt8Calibrator(calib.get());
    prefix = "INT8";
  } else {
    cout << "Use FP32 or TF32 Mode\n";
    config->setFlag(BuilderFlag::kTF32);
    prefix = "FP32";
  }

  // build engine
  auto engine = hackathon::UniquePtr<ICudaEngine>(
      builder->buildEngineWithConfig(*network, *config));
  IHostMemory* trt_model_stream = engine->serialize();
  if (!trt_model_stream) {
    cout << "serialize model failure\n";
    return false;
  }

  string save_trt_file(prefix + "_" + kTrtPlanName);
  ofstream ofs(save_trt_file);
  ofs.write(static_cast<const char*>(trt_model_stream->data()),
            trt_model_stream->size());
  ofs.close();
  cout << "Save TensorRT Model:" + save_trt_file + " Success\n";

  if (trt_model_stream) {
    trt_model_stream->destroy();
  }

  return true;
}

bool parseArgs(int argc, char* argv[], hackathon::CommandParam& command_param) {
  if (argc == 1) {
    printf("TensorRT2022 Tool\n");
    exit(0);
  }

  if (argc < 4) {
    printf("\n");
    printf("Mandatory params:\n");
    printf("  [onnx path]   : onnx model path\n");
    printf("  [enable fp16] : enable fp16(only 1 or 0)\n");
    printf("  [enable int8] : enable int8(only 1 or 0)\n");
    return false;
  }

  command_param.model_path = std::string(argv[1]);
  command_param.use_half = std::atoi(argv[2]) == 1 ? true : false;
  command_param.use_int8 = std::atoi(argv[3]) == 1 ? true : false;
  if (hackathon::check_file_exist(command_param.model_path) == false) {
    printf("onnx path: %s not exist\n", command_param.model_path.c_str());
    return false;
  }

  return true;
}

int main(int argc, char* argv[]) {
  hackathon::CommandParam command_param;
  bool r = parseArgs(argc, argv, command_param);
  if (!r) {
    cout << "Parse Command Failure\n";
    return -1;
  }

  initLibNvInferPlugins(&gLogger.getTRTLogger(), "");
  if (!OnnxToTrt(command_param)) {
    cout << "Onnx TensorRT Convert Failure\n";
    return -1;
  }

  cout << "Onnx TensorRT Convert Success" << endl;

  return 0;
}
