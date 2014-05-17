#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#define vectorSize 1000

float totalTime;

cudaError_t parallelWithCuda(double *a, double *b, double *c, unsigned int size);

__global__ void parallelVM(double *a, double *b, double *c)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	c[i] = a[i] * b[i];
}

int main()
{
    int i;
	double k = 5.0;
	double j = (float)vectorSize/2.0;

	double a[vectorSize];
	double b[vectorSize]; 
	double c[vectorSize]; 

	// fill vectors with random values
	for ( i = 0; i < vectorSize; i++ ){
				a[i] = (double)rand()/ (double)RAND_MAX;
				b[i] = (double)rand()/ (double)RAND_MAX;
				c[i] = 0.0;
	}

    // Multiply vectors
    cudaError_t cudaStatus = parallelWithCuda(a, b, c, vectorSize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "seqWithCuda failed!");
        return 1;
    }
	/*
	// display vectors for debugging
	for ( i = 0; i < vectorSize; i++ ){
				printf("\n%f * %f = %f", a[i], b[i], c[i]);
	}
	*/

	printf("\n\nTotal time: %.3fms\n\n", totalTime);
	
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaDeviceReset();

    return 0;
}

// Helper function for using CUDA to multiply vectors
cudaError_t parallelWithCuda(double *a, double *b, double *c, unsigned int size)
{
    double *dev_a;
    double *dev_b;
    double *dev_c;
    cudaError_t cudaStatus;
	cudaError_t error;
	cudaEvent_t start, stop;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto End;
    }

    // Allocate GPU buffers for three vectors (two input, one output)  

    cudaMalloc((void**)&dev_a, size * sizeof(double));
	cudaMalloc((void**)&dev_b, size * sizeof(double));
	cudaMalloc((void**)&dev_c, size * sizeof(double));

    // Copy input vectors from host memory to GPU buffers.
    cudaMemcpy(dev_a, a, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size * sizeof(double), cudaMemcpyHostToDevice);

	// Allocate CUDA events that we'll use for timing
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start, NULL);

	// multiply the vectors
	parallelVM<<< vectorSize/512, 512 >>>(dev_a, dev_b, dev_c);

	// Record the stop event
    cudaEventRecord(stop, NULL);
    // Wait for the stop event to complete
    cudaEventSynchronize(stop);

	totalTime = 0.0f;
	cudaEventElapsedTime(&totalTime, start, stop);

    cudaDeviceSynchronize();

    // Copy output vector from GPU buffer to host memory.
    cudaMemcpy(c, dev_c, size * sizeof(double), cudaMemcpyDeviceToHost);

End:
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    
    return cudaStatus;
}
