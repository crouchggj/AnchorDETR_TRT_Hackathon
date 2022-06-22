#include <assert.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

static const float HALF_FLT_MAX = 65504.F;
#define FINAL_MASK 0xffffffff

template <typename T>
inline __device__ T ldg(const T* val) {
  return __ldg(val);
}

template <typename T>
__inline__ __device__ T warpReduceSum(T val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  return val;
}

/* Calculate the sum of all elements in a block */
template <typename T>
__inline__ __device__ T blockReduceSum(T val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceSum<T>(val);

  if (lane == 0) shared[wid] = val;

  __syncthreads();

  // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
  // blockDim.x is not divided by 32
  val = (threadIdx.x < (blockDim.x / 32.f)) ? shared[lane] : (T)(0.0f);
  val = warpReduceSum<T>(val);

  return val;
}

template <typename T>
__global__ void generalLayerNorm(const T* __restrict input,
                                 const T* __restrict gamma,
                                 const T* __restrict beta, T* output, int m,
                                 int n) {
  const int tid = threadIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean = 0.0f;
  float variance = 0.0f;

  float local_sum = 0.0f;
  for (int i = tid; i < n; i += blockDim.x) {
    local_sum += (float)(ldg(&input[blockIdx.x * n + i]));
  }

  mean = blockReduceSum(local_sum);

  if (threadIdx.x == 0) {
    s_mean = mean / n;
  }
  __syncthreads();

  float local_var_sum = 0.0f;
  for (int i = tid; i < n; i += blockDim.x) {
    float diff = (float)(ldg(&input[blockIdx.x * n + i])) - s_mean;
    local_var_sum += diff * diff;
  }
  variance = blockReduceSum(local_var_sum);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / n + 1e-6f);
  }
  __syncthreads();

  for (int i = tid; i < n; i += blockDim.x) {
    float beta_val = (beta == nullptr) ? 0.0f : (float)ldg(&beta[i]);
    output[blockIdx.x * n + i] =
        (T)((((float)input[blockIdx.x * n + i] - s_mean) * s_variance) *
                (float)(ldg(&gamma[i])) +
            beta_val);
  }
}

template <typename T>
void invokeGeneralLayerNorm(T* out, const T* input, const T* gamma,
                            const T* beta, const int m, const int n,
                            cudaStream_t stream, int opt_version) {
  dim3 grid(m);
  dim3 block(n);
  generalLayerNorm<T>
      <<<grid, block, 0, stream>>>(input, gamma, beta, out, m, n);
}
