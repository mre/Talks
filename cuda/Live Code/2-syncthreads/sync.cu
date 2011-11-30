#include <stdio.h>

#define N 16

void print(unsigned int* data) {
  int i;
  for (i = 0; i < N; ++i) {
    printf("%d", data[i]);
  }
}
__global__ void kernel(unsigned int *dev_data)
{
	int x = threadIdx.x;
  dev_data[x] = 1;
/*	__syncthreads(); */
  //dev_data[x] += dev_data[N-x];
}

int main( void )
{
  unsigned int data[N] = {0};
	unsigned int *dev_data;
  size_t size = N*sizeof(unsigned int);
	cudaMalloc((void **) &dev_data, size);
	cudaMemcpy(dev_data, data, size, cudaMemcpyHostToDevice);
	
	dim3 threads(16, 0);
	kernel<<<1, threads>>>(dev_data);

	cudaMemcpy(data, dev_data, size, cudaMemcpyDeviceToHost);
	cudaFree(dev_data);
  print(data);
	return 0;
}
