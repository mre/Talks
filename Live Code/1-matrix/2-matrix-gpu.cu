#include <stdlib.h>
#include <stdio.h>

#define DIM 4

/* CUDA memory allocation and kernel in .cu File */
__global__ void device_mult_matrix(float *Md, float *Nd, float *Pd)
{
  /* Thread ID */
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  /* Compute one value per thread */
  float v = 0;
  int k;
  for (k = 0; k < DIM; ++k) {
    float Md_value = Md[ty * DIM + k];
    float Nd_value = Nd[k * DIM + tx];
    v += Md_value * Nd_value;
  }
  /* Write to result matrix on device */
  Pd[ty * DIM + tx] = v;
}

void mult_matrix(float *M, float *N, float *P) 
{
  float *Md, *Nd, *Pd;
  size_t size = DIM*DIM * sizeof(float);

  /* Allocate device memory */
  cudaMalloc((void**) &Md, size);
  cudaMalloc((void**) &Nd, size);
  cudaMalloc((void**) &Pd, size);

  /* Copy matrices to device */
  cudaMemcpy(Md, M, size, cudaMemcpyHostToDevice);
  cudaMemcpy(Nd, N, size, cudaMemcpyHostToDevice);

  /* Perform calculation */
  dim3 dimBlock(DIM, DIM);
  dim3 dimGrid(1,1);
  device_mult_matrix<<<dimGrid, dimBlock>>>(Md, Nd, Pd);

  /* Copy P from device to host and free matrices */
  cudaMemcpy(P, Pd, size, cudaMemcpyDeviceToHost);
  cudaFree(Md); cudaFree(Nd); cudaFree(Pd);
}

void print_matrix(float *P)
{
  int i, j;
  for (i = 0; i < DIM; ++i) {
    for (j = 0; j < DIM; ++j) {
      printf("%f ", P[i*DIM + j]);
    }
    printf("\n");
  }
}

int main(void)
{
  /* Initialize matrices */
  float M[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
  float N[] = {42,5,13,23,4,11,9,21,85,7,8,1,49,3,2,1};
  float *P = (float*) malloc(DIM*DIM* sizeof(float)); /* Result */

  mult_matrix(M, N, P);
  print_matrix(P);
  return 0;
}

