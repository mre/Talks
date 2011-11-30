#include <stdlib.h>
#include <stdio.h>

#define N 2 /* Number of threads */

void print(int *cells) 
{
  int i;
  for (i = 0; i < N; ++i) {
    printf("%d ", cells[i]);
  }
  printf("\n");
}

__device__ void step(int *dev_cells)
{
  int i, cell;
  i = threadIdx.x;

  cell = (i == 0 ? 1 : 0);
  dev_cells[i] += dev_cells[cell];
}

__global__ void cuda_inc(int *dev_cells)
{
  int i;
  for (i=0; i < 1000; ++i) {
    step(dev_cells);
  }
}

int main(void)
{
  int cells[N] = {1, 1};

  /* Copy cells to device */
  int *dev_cells;
  size_t size = N * sizeof(int);
  cudaMalloc((void**) &dev_cells, size);
  cudaMemcpy(dev_cells, cells, size, cudaMemcpyHostToDevice);

  cuda_inc<<<1,N>>>(dev_cells); /* Inplace */

  /* Get cells from device */
  cudaMemcpy(cells, dev_cells, size, cudaMemcpyDeviceToHost);
  cudaFree(dev_cells);
  print(cells);
}
