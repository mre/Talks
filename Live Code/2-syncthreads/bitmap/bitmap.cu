#define _USE_MATH_DEFINES
#include <math.h>
#include "../common/glBitmap.h"
#include "../common/common.h"

#define DIM 1024

__global__ void kernel(unsigned char *ptr)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int offset = x + y * blockDim.x * gridDim.x;
	
	__shared__ float shared[16][16];

	const float period = 128.0f;
	shared[threadIdx.x][threadIdx.y] =
		255 * (sinf(x * 2.0f * M_PI/ period) + 1.0f) *
		(sinf(y * 2.0f * M_PI / period) + 1.0f) / 4.0f;

/*	__syncthreads(); */

	ptr[offset*4 + 0] = 0;
	ptr[offset*4 + 1] = shared[15-threadIdx.x][15-threadIdx.y];
	ptr[offset*4 + 2] = 0;
	ptr[offset*4 + 3] = 255;
}

int main( void )
{
	GlBitmap bitmap(DIM, DIM);
	unsigned char *dev_data;
	dim3 grid(DIM/16, DIM/16);
	dim3 threads(16, 16);

	SAFE_CALL( cudaMalloc(&dev_data, bitmap.get_size()));
	
	kernel<<<grid, threads>>>(dev_data);

	SAFE_CALL( cudaMemcpy(bitmap.get_data(), dev_data, bitmap.get_size(), cudaMemcpyDeviceToHost));
	cudaFree(dev_data);

	bitmap.display_and_exit();
	return 0;
}
