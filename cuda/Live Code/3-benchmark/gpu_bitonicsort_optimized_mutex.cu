/*
 * Parallel bitonic sort using CUDA.
 * Based on http://www.tools-of-computing.com/tc/CS/Sorts/bitonic_sort.htm
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

/*
 * Every thread gets exactly one value in the unsorted array.
 */
#define THREADS 512
#define BLOCKS 65536
#define NUM_VALS THREADS*BLOCKS

float random_float()
{
  return (float)rand()/(float)RAND_MAX;
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

/*
 * GPU simple synchronization function.
 * See: http://eprints.cs.vt.edu/archive/00001087/01/TR_GPU_synchronization.pdf
 */

/* The mutex variable */
__device__ int g_mutex = 0;

__device__ void __gpu_sync(int goalVal)
{
  /* Thread ID in a block */
  int tid_in_block = threadIdx.x * blockDim.y + threadIdx.y;

  /* Only thread 0 is used for synchronization */
  if (tid_in_block == 0) {
    atomicAdd(&g_mutex, 1);

    while(g_mutex != goalVal) {
      /* Wait until all blocks have increased g_mutex */
    }
  }
  __syncthreads();
}

__device__ void bitonic_sort_step(float *dev_values, int j, int k)
{
  unsigned int i, ixj; /* Sorting partners: i and ixj */
  i = threadIdx.x + blockDim.x * blockIdx.x;
  ixj = i^j;

  /* The threads with the lowest ids sort the array. */
  if ((ixj)>i) {
    if ((i&k)==0) {
      /* Sort ascending */
      if (dev_values[i]>dev_values[ixj]) {
        /* exchange(i,ixj); */
        float temp = dev_values[i];
        dev_values[i] = dev_values[ixj];
        dev_values[ixj] = temp;
      }
    }
    if ((i&k)!=0) {
      /* Sort descending */
      if (dev_values[i]<dev_values[ixj]) {
        /* exchange(i,ixj); */
        float temp = dev_values[i];
        dev_values[i] = dev_values[ixj];
        dev_values[ixj] = temp;
      }
    }
  }
}

__global__ void bitonic_sort(float *dev_values)
{
  int j, k, goal_value = 0;
  /* Major step */
  for (k = 2; k <= NUM_VALS; k <<= 1) {
    /* Minor step */
    for (j=k>>1; j>0; j=j>>1) {
      bitonic_sort_step(dev_values, j, k);
      goal_value += BLOCKS;
      __gpu_sync(goal_value);
    }
  }
}

int main(void)
{
  float *values = (float*) malloc( NUM_VALS * sizeof(float));
  array_fill(values, NUM_VALS);
  /* array_print(values, NUM_VALS); */

  float *dev_values;

  size_t size = NUM_VALS * sizeof(float);
  cudaMalloc((void**) &dev_values, size);
  cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);

  dim3 blocks(BLOCKS,1); /* Number of blocks */
  dim3 threads(THREADS,1); /* Number of threads */
  bitonic_sort<<<blocks, threads>>>(dev_values);

  cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
  cudaFree(dev_values);

  /*array_print(values, NUM_VALS);*/
}
