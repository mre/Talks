!SLIDE

# Beispiel: Matrixprodukt

![Cube](Grafiken/matrixmult1.png)

<!SLIDE transition=fade>

# Beispiel: Matrixprodukt

![Cube](Grafiken/matrixmult2.png)

!SLIDE

## Implementierung auf CPU

    @@@ C
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

!SLIDE

## Parallelisierung mit CUDA

    @@@ C
    void mult_matrix(float *M, float *N, float *P)
    {


      /*
       * TODO:
       * - Allocate device memory
       * - Copy matrices to device
       * - Perform calculation
       * - Copy result and free matrices
       */



    }

!SLIDE transition=fade

## Parallelisierung mit CUDA

    @@@ C
    void mult_matrix(float *M, float *N, float *P)
    {
      /* Allocate device memory */
      float *Md, *Nd, *Pd;
      size_t size = DIM*DIM * sizeof(float);
      cudaMalloc((void**) &Md, size);
      cudaMalloc((void**) &Nd, size);
      cudaMalloc((void**) &Pd, size);


      /* Copy matrices to device */
      /* Perform calculation */
      /* Copy result from device to host and free matrices */

    }

!SLIDE transition=fade

## Parallelisierung mit CUDA

    @@@ C
    void mult_matrix(float *M, float *N, float *P)
    {
      /* Allocate device memory... */


      /* Copy matrices to device */
      cudaMemcpy(Md, M, size, cudaMemcpyHostToDevice);
      cudaMemcpy(Nd, N, size, cudaMemcpyHostToDevice);


      /* Perform calculation */
      /* Copy result from device to host and free matrices */


    }

!SLIDE transition=fade

## Parallelisierung mit CUDA

    @@@ C
    void mult_matrix(float *M, float *N, float *P)
    {
      /* Allocate device memory... */
      /* Copy matrices to device... */

      /* Perform calculation */
      dim3 dimBlock(DIM, DIM);
      dim3 dimGrid(1,1);
      device_mult_matrix<<<dimGrid, dimBlock>>>(Md, Nd, Pd);



      /* Copy result from device to host and free matrices */

    }

!SLIDE

## Parallelisierung mit CUDA

    @@@ C
    __global__ void device_mult_matrix(
      float *Md, float *Nd, float *Pd)
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


!SLIDE

## Parallelisierung mit CUDA

    @@@ C
    void mult_matrix(float *M, float *N, float *P)
    {
      /* Allocate device memory... */
      /* Copy matrices to device... */
      /* Perform calculation... */

      /* Copy result from device to host */
      cudaMemcpy(P, Pd, size, cudaMemcpyDeviceToHost);


      /*free matrices */
      cudaFree(Md);
      cudaFree(Nd);
      cudaFree(Pd);
    }
