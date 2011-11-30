/*
 * Bitonic Sort
 * ============
 *
 * Simple CPU implementation.
 *
 * Derived from:
 * http://www.tools-of-computing.com/tc/CS/Sorts/bitonic_sort.htm
 * http://forums.nvidia.com/index.php?showtopic=84651
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

//#define ELEMENTS 2*1024*1024 /* Must be a power of 2 */
//#define ELEMENTS 16
//#define ELEMENTS 32
//#define ELEMENTS 64
//#define ELEMENTS 128
//#define ELEMENTS 512
//#define ELEMENTS 1024
//#define ELEMENTS 2048
//#define ELEMENTS 4096
//#define ELEMENTS 524288
//#define ELEMENTS 1048576
//#define ELEMENTS 2097152
#define ELEMENTS 4194304
//#define ELEMENTS 33554432

/**
 * Generate a pseudo random float value
 */
float random_float() 
{
  float r = (float)rand()/(float)RAND_MAX;
  return r;
}

void bitonic_sort(float *values)
{
  int i,j,k;
  for (k=2;k<=ELEMENTS;k=2*k) {
    for (j=k>>1;j>0;j=j>>1) {
      for (i=0;i<ELEMENTS;++i) {
        int ixj=i^j;
        if ((ixj)>i) {
          if ((i&k)==0 && values[i]>values[ixj]) {
            // exchange(i,ixj);
            float temp = values[i];
            values[i] = values[ixj];
            values[ixj] = temp;
          }
          if ((i&k)!=0 && values[i]<values[ixj]) {
            // exchange(i,ixj);
            float temp = values[i];
            values[i] = values[ixj];
            values[ixj] = temp;
          }
        }
      }
    }
  }
}

/**
 * Print an array
 */
void array_print(float *arr, size_t size)
{
  size_t i;
  for (i = 0; i < size; ++i) {
    printf("%1.3f ", arr[i]);
  }
  printf("\n");
}

/**
 * Fill an array with random floating-point values
 */
void array_fill(float *arr, size_t size) 
{
  srand(time(NULL));
  size_t i;
  for (i = 0; i < size; ++i) {
    arr[i] = random_float();
  }
}

void print_elapsed(clock_t start, clock_t stop)
{
  double elapsed = ((double) (stop - start)) / CLOCKS_PER_SEC;
  printf("Elapsed time: %.3fs\n", elapsed);
}

int main(void)
{
  clock_t start, stop;

  /* Generate a large number of floating-point values */
  float *arr = (float *) malloc(ELEMENTS * sizeof(float));
  array_fill(arr, ELEMENTS);
  //array_print(arr, ELEMENTS);

  start = clock();
  bitonic_sort(arr);
  stop = clock();

  print_elapsed(start, stop);
  array_print(arr, ELEMENTS);
}
