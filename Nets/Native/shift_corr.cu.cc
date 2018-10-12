#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "stdio.h"

#define CUDA_NUM_THREADS 512

#define WARPS_PER_BLOCK 1
#define THREADS_PER_WARP 32

inline int GET_BLOCKS(const int N) {
    return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

__global__ void CorrelateData(const int nthreads, int num, int topwidth, int topheight, int topchannels, int topcount,
  int max_displacement, int x_shift,
  int bottomwidth, int bottomheight, int bottomchannels,
  const float *bottom0, const float *bottom1, float*top)
{
  extern __shared__ char patch_data_char[];

  float *patch_data = (float *)patch_data_char;

    // First (upper left) position of kernel upper-left corner in current center position of neighborhood in image 1
  int x1 = blockIdx.x + max_displacement;
  int y1 = blockIdx.y;
  int item = blockIdx.z;
  int ch_off = threadIdx.x;

  for(int ch = ch_off; ch < bottomchannels; ch += (WARPS_PER_BLOCK*THREADS_PER_WARP)) { // CHANNELS
      int idx1 = ((item * bottomheight + y1) * bottomwidth + x1) * bottomchannels + ch;
      int idxPatchData = ch;
      patch_data[idxPatchData] = bottom0[idx1];
  }

  __syncthreads();

  __shared__ float sum[WARPS_PER_BLOCK*THREADS_PER_WARP];

  // Compute correlation
  for(int top_channel = 0; top_channel < topchannels; top_channel++) {
    sum[ch_off] = 0;

    int s2o = (top_channel % topchannels + x_shift);

    for(int ch = ch_off; ch < bottomchannels; ch += (WARPS_PER_BLOCK*THREADS_PER_WARP)) { // CHANNELS
      int x2 = x1 + s2o;

      int idxPatchData = ch;
      int idx2 = ((item * bottomheight + y1) * bottomwidth + x2) * bottomchannels + ch;

      sum[ch_off] += patch_data[idxPatchData] * bottom1[idx2];

    }

    __syncthreads();

    if(ch_off == 0) {
        float total_sum = 0;
        for(int idx = 0; idx < WARPS_PER_BLOCK*THREADS_PER_WARP; idx++) {
            total_sum += sum[idx];
        }
        const int sumelems = bottomchannels;
        const int index = ((top_channel*topheight + blockIdx.y)*topwidth)+blockIdx.x;
        top[index + item*topcount] = total_sum / (float)sumelems;
    }
  }
}

// == Correlation Backward Pass Kernel (For Input 0)
__global__ void CorrelateDataBackward0(const int nthreads, int item, int topchannels,
  int max_disp, int x_shift,
  int width, int height, int pbottomwidth, int bottomchannels, int bottomcount,
  float *bottom0diff, const float *bottom1, const float *topdiff)
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int n = index % bottomchannels; //channels
    int l = (index / bottomchannels) % pbottomwidth + max_disp; //w-pos
    int m = (index / bottomchannels / pbottomwidth) % height; //h-pos

    //Get X,Y ranges and clamp

    int xmin = l - max_disp;
    int ymin = m;

    // Same here:
    int xmax = l - max_disp;
    int ymax = m;

    float sum = 0;
    if(xmax>=0 && ymax>=0 && (xmin<=width-1) && (ymin<=height-1))
    {
        xmin = max(0,xmin);
        xmax = min(width-1,xmax);

        ymin = max(0,ymin);
        ymax = min(height-1,ymax);

        {
          for(int o = x_shift; o < x_shift + topchannels; o++) {

            // Get bottom1 data:
            int idxbot1 = ((item * height + m) * pbottomwidth + (l+o)) * bottomchannels + n;

            float bot1tmp = bottom1[idxbot1]; // bottom1[l+s2o,m,n]
            /*if (bot1tmp > 0) {
                printf("BOTTOM1 > 0: index: %d, item: %d, n: %d, l: %d, m: %d, o: %d, idxbot1: %d, xmin: %d, xmax: %d, ymin: %d, ymax: %d \n\n", index, item, n, l, m, o, idxbot1, xmin, xmax, ymin, ymax);
            }*/

            // Index offset for topdiff in following loops:
            int op = (o-x_shift); // index [o,p]
            int idxopoffset = (item * topchannels + op);

            for(int y = ymin; y <= ymax; y++) {
              for(int x = xmin; x <= xmax; x++) {
                int idxtopdiff = (idxopoffset * height + y) * width + x; // topdiff[x,y,o,p]
                sum += topdiff[idxtopdiff] * bot1tmp;
              /*  if (bot1tmp > 0) {
                    printf("SUM + > 0: index: %d, item: %d, n: %d, l: %d, m: %d, o: %d, idxbot1: %d, x_shift: %d, idxopoffset: %d, idxtopdiff: %d \n\n", index, item, n, l, m, o, idxbot1, x_shift, idxopoffset, idxtopdiff);
                }*/
              }
            }
          }
        }
    }
    const int sumelems = bottomchannels;
    const int bot0index = ((n * height) + m) * pbottomwidth + l;
    bottom0diff[bot0index + item*bottomcount] = sum / (float)sumelems;
  }

}

// == Correlation Backward Pass Kernel (For Input 1)
__global__ void CorrelateDataBackward1(const int nthreads, int item, int topchannels,
  int max_disp, int x_shift,
  int width, int height, int pbottomwidth, int bottomchannels, int bottomcount,
  const float *bottom0, float *bottom1diff, const float *topdiff)
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int n = index % bottomchannels; //channels
    int l = (index / bottomchannels) % pbottomwidth + max_disp; //w-pos
    int m = (index / bottomchannels / pbottomwidth) % height; //h-pos

    float sum = 0;
    {

      for(int o = x_shift; o < x_shift + topchannels; o++) {

        //Get X,Y ranges and clamp
        int xmin = l - max_disp - o;
        int ymin = m;

        // Same here:
        int xmax = l - max_disp - o;
        int ymax = m;

        if(xmax>=0 && ymax>=0 && (xmin<=width-1) && (ymin<=height-1))
        {
            xmin = max(0,xmin);
            xmax = min(width-1,xmax);

            ymin = max(0,ymin);
            ymax = min(height-1,ymax);

            // Get bottom0 data:
            int idxbot0 = ((item * height + m) * pbottomwidth + (l-o)) * bottomchannels + n;
            float bot0tmp = bottom0[idxbot0]; // bottom1[l+o,m,n]
            /*if (bot0tmp > 0) {
                printf("BOTTOM0 > 0: index: %d, item: %d, n: %d, l: %d, m: %d, o: %d, idxbot0: %d, xmin: %d, xmax: %d, ymin: %d, ymax: %d \n\n", index, item, n, l, m, o, idxbot0, xmin, xmax, ymin, ymax);
            }*/
            // Index offset for topdiff in following loops:
            int op = (o-x_shift); // index [o,p]
            int idxOpOffset = (item * topchannels + op);

            for(int y = ymin; y <= ymax; y++) {
              for(int x = xmin; x <= xmax; x++) {
                int idxtopdiff = (idxOpOffset * height + y) * width + x; // topdiff[x,y,o,p]
                sum += topdiff[idxtopdiff] * bot0tmp;
              }
            }
        }
      }
    }
    const int sumelems = bottomchannels;
    const int bot1index = ((n * height) + m) * pbottomwidth + l;
    bottom1diff[bot1index + item*bottomcount] = sum / (float)sumelems;
  }

}

void ShiftCorrKernelLauncher(const float *values0, const float *values1, const int max_disp,
        const int batch_size, const int in_h, const int in_w, const int in_channels, float *out) {

    const int single_direction_ = 0;

    int top_channels_;
    if(single_direction_ != 0) {
      top_channels_ = max_disp + 1;
    } else {
      top_channels_ = max_disp * 2 + 1;
    }

    const int top_height_ = in_h;
    const int top_width_ = in_w - 2*max_disp;
    const int paddedwidth = in_w;
    const int topcount = top_height_ * top_width_ * top_channels_;

    dim3 threadsPerBlock(THREADS_PER_WARP * WARPS_PER_BLOCK);

    const int shared_memory_per_block = in_channels;

    int x_shift = - max_disp;
    if(single_direction_ == -1) { // to the left
      x_shift = -top_channels_;
    } else if(single_direction_ == 1) { // to the right
      x_shift = 0;
    }

    // Correlation1DLayer
    int topThreadCount = topcount;

    dim3 totalBlocksCorr(top_width_, top_height_, batch_size);

    CorrelateData<<<totalBlocksCorr, threadsPerBlock, shared_memory_per_block * sizeof(float)>>>(
        topThreadCount,
        batch_size, top_width_, top_height_, top_channels_, topcount,
        max_disp, x_shift,
        paddedwidth, in_h, in_channels,
        values0, values1, out
        );
}

void ShiftCorrGradKernelLauncher(const float *input0, const float *input1, const float *grad,
        const int max_disp, const int batch_size, const int height, const int paddedwidth,
        const int channels, float *output0, float *output1)
{
    // Get top diff, compute bottom diff

    const int width = paddedwidth - 2*max_disp;
    const int bottomcount = channels * height * paddedwidth;

    int botThreadCount = bottomcount;

    // CorrelationLayerBackward

    const int single_direction_ = 0;

    int top_channels_;
    if(single_direction_ != 0) {
      top_channels_ = max_disp + 1;
    } else {
      top_channels_ = max_disp * 2 + 1;
    }

    int x_shift = - max_disp;
    if(single_direction_ == -1) { // to the left
      x_shift = - top_channels_;
    } else if(single_direction_ == 1) { // to the right
      x_shift = 0;
    }

    // == Run kernel Backward 0
    for(int n = 0; n < batch_size; n++) {
    //Bottom0:
    CorrelateDataBackward0<<<GET_BLOCKS(botThreadCount), CUDA_NUM_THREADS>>>(
        botThreadCount,
        n, top_channels_,
        max_disp, x_shift,
        width, height, paddedwidth, channels, bottomcount,
        output0, input1, grad
        );

    }

    // == Run kernel Backward 1
    for(int n = 0; n < batch_size; n++) {
    CorrelateDataBackward1<<<GET_BLOCKS(botThreadCount), CUDA_NUM_THREADS>>>(
        botThreadCount,
        n, top_channels_,
        max_disp, x_shift,
        width, height, paddedwidth, channels, bottomcount,
        input0, output1, grad
        );

    }

}
#endif
