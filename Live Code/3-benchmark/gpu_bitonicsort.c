/*
 * Parallel bitonic sort using CUDA.
 * Based on http://www.tools-of-computing.com/tc/CS/Sorts/bitonic_sort.htm
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "cuPrintf.cu"


/*
 * Every thread gets exactly one value in the unsorted array.
 */
#define NUM_VALS 512

float random_float()
{
  float r = (float)rand()/(float)RAND_MAX;
  return r;
}

void array_print(float *arr, int length) 
{
  int i;
  for (i = 0; i < length; ++i) {
    printf("%1.3f ",  arr[i]);
  }
  printf("\n");
}

void array_fill(float *arr, int length)
{
  srand(time(NULL));
  int i;
  for (i = 0; i < length; ++i) {
    arr[i] = random_float();
  }
}

__device__ void sort_step(float *dev_values, int j, int k)
{
  /* Sorting partners: i and ixj */
  unsigned int i, ixj;
  i = threadIdx.x;
  ixj = i^j;

  /* The threads with the lowest ids sort the array. */
  if ((ixj)>i){
    if ((i&k)==0 && dev_values[i]>dev_values[ixj]) {
      //exchange(i,ixj);
      float temp = dev_values[i];
      dev_values[i] = dev_values[ixj];
      dev_values[ixj] = temp;
    }
    if ((i&k)!=0 && dev_values[i]<dev_values[ixj]){
      // exchange(i,ixj);
      float temp = dev_values[i];
      dev_values[i] = dev_values[ixj];
      dev_values[ixj] = temp;
    }
  }
}

__global__ void bitonic_sort(float *dev_values)
{
  int j, k;
  /* Major step */
  for (k = 2; k <= NUM_VALS; k <<= 1) {
    /* Minor step */
    for (j=k>>1; j>0; j=j>>1) {
      sort_step(dev_values, j, k);
      __syncthreads();
    }
  }
}

int main(void)
{
  float *values = (float*) malloc( NUM_VALS * sizeof(float));
  array_fill(values, NUM_VALS);
  array_print(values, NUM_VALS);

  float *dev_values;

  size_t size = NUM_VALS * sizeof(float);
  cudaMalloc((void**) &dev_values, size);
  cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);

  dim3 blocks(1,1); /* Number of blocks */
  dim3 threads(NUM_VALS,1); /* Number of threads */
  bitonic_sort<<<blocks, threads>>>(dev_values);

  cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
  cudaFree(dev_values);

  array_print(values, NUM_VALS);
}
