echo "Start to build trt_hackathon image..."
# TRT download url: https://developer.nvidia.com/nvidia-tensorrt-8x-download
tar zxf TensorRT-8.4.0.6.Linux.x86_64-gnu.cuda-11.0-~11.6.cudnn8.3.tar.gz
echo "Tar TensorRT Done."
docker build -t trt_hackathon:v2 .
echo "Build Image Success."


