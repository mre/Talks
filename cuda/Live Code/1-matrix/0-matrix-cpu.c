#include <stdlib.h>
#include <stdio.h>

#define DIM 4
#define ELEMENTS (DIM)*(DIM)

void mult_matrix(float *M, float *N, float *P) 
{
  int i, j, k;
  for (i = 0; i < DIM; ++i) {
    for (j = 0; j < DIM; ++j) {
      float sum = 0;
      for (k = 0; k < DIM; ++k) {
        float a = M[i*DIM + k];
        float b = N[k*DIM + j];
        sum += a * b;
      }
      P[i*DIM + j] = sum;
    }
  }
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
  float M[ELEMENTS] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
  float N[ELEMENTS] = {42,5,13,23,4,11,9,21,85,7,8,1,49,3,2,1};
  float *P = (float*) malloc(ELEMENTS * sizeof(float)); /* Result */

  mult_matrix(M, N, P);
  print_matrix(P);
  return 0;
}

