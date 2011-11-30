#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define ELEMENTS 1000000

/**
 * Generate a pseudo random float value
 */
float random_float() 
{
  float r = (float)rand()/(float)RAND_MAX;
  return r;
}

void array_mergesort(float *arr, size_t size)
{
 if(size > 1){
     float half1[size/2];
     float half2[(size + 1)/2];
     size_t i;

     for(i = 0; i < size/2; ++i)
         half1[i] = arr[i];
     for(i = size/2; i < size; ++i)
         half2[i - size/2] = arr[i];
     array_mergesort(half1,size/2);
     array_mergesort(half2,(size + 1)/2);

     float *pos1 = &half1[0];
     float *pos2 = &half2[0];
     for(i = 0; i < size; ++i){
         if(*pos1 <= *pos2){
             arr[i] = *pos1;
             if(*pos1 == half1[size/2 - 1]){
                 pos1 = &half2[(size+1)/2 - 1];
             }
             else{
                 ++pos1;
             }
         }
         else{
             arr[i] = *pos2;
             if(*pos2 == half2[(size + 1)/2 - 1]){
                 pos2 = &half1[size/2 - 1];
             }
             else{
                 ++pos2;
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
    printf("%f ", arr[i]);
  }
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

int main(void)
{
  /* Generate a large number of floating-point values */
  float *arr = (float *) malloc(ELEMENTS * sizeof(float));
  array_fill(arr, ELEMENTS);
  /* array_print(arr, ELEMENTS); */
  array_mergesort(arr, ELEMENTS);
  array_print(arr, ELEMENTS);
}
