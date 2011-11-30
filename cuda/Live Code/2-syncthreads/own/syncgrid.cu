#include <stdlib.h>
#include <stdio.h>

void print(int *grid) 
{
  int i;
  for (i = 0; i < 4; ++i) {
    printf("%d ", grid[i]);
  }
  printf("\n");
}

__device__ void step(int *dev_grid, int j)
{
  int i = threadIdx.x;
  int ixj = i^j;

  /* The threads with the lowest ids sort the array. */
  if ((ixj)>i){
    int temp = dev_grid[i];
    dev_grid[i] = dev_grid[ixj];
    dev_grid[ixj] = temp;
  }
}

__global__ void cuda_swap(int *dev_grid)
{
  int j, z;
  for (z=0; z< 200000; ++z) {
    for (j=2; j>0; j=j>>1) {
      step(dev_grid, j);
      //__syncthreads();
    }
  }
}

int main(void)
{
  int grid[4] = {3,2,1,0};

  /* Copy grid to device */
  int *dev_grid;
  size_t size = 4 * sizeof(int);
  cudaMalloc((void**) &dev_grid, size);
  cudaMemcpy(dev_grid, grid, size, cudaMemcpyHostToDevice);

  cuda_swap<<<1,4>>>(dev_grid); /* Inplace */

  /* Get grid from device */
  cudaMemcpy(grid, dev_grid, size, cudaMemcpyDeviceToHost);
  cudaFree(dev_grid);

  print(grid);
}
