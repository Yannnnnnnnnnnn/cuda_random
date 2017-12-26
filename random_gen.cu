#include "cuda_runtime.h"
#include "device_launch_parameters.h" 
#include "curand_kernel.h" 

#include <stdio.h>
#include <time.h>


__global__ void kernel_set_random(curandState *curand_states,int size,long clock_for_rand)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    
    if(x<0 || x>size)
    {
        return;
    }

    curand_init(clock_for_rand,x,0,&curand_states[x]);
}

__global__ void kernel_random(float *dev_random_array,int height,int width,curandState *curand_states)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;

    if(x<0 || x>width || y<0 || y>height)
    {
        return;
    }

    int pos = y*width + x;

    dev_random_array[pos] = abs(curand_uniform(&curand_states[x]));
}


int main()
{
    const int array_size_width = 1000;
    const int array_size_height = 1000;
    float random_array[array_size_width*array_size_height];
    for(int i=0;i<array_size_width*array_size_height;i++)
    {
        random_array[i] = 0;
    }

    //error status
    cudaError_t cuda_status;

    //only chose one GPU
    cuda_status = cudaSetDevice(0);
    if(cuda_status != cudaSuccess)
    {
        fprintf(stderr,"cudaSetDevice failed! Do you have a CUDA-Capable GPU installed?");
        return 0;
    }

    float *dev_random_array;
    curandState *dev_states;

     //allocate memory on the GPU
    cuda_status = cudaMalloc((void**)&dev_random_array,sizeof(float)*array_size_width*array_size_height);
    if(cuda_status != cudaSuccess)
    {
        fprintf(stderr,"dev_reference_image cudaMalloc Failed");
        exit( EXIT_FAILURE );
    }
    cuda_status = cudaMalloc((void **)&dev_states,sizeof(curandState)*array_size_width);
    if(cuda_status != cudaSuccess)
    {
        fprintf(stderr,"dev_states cudaMalloc Failed");
        exit( EXIT_FAILURE );
    }

    long clock_for_rand = clock();

    dim3 threads_x(16,1);
    dim3 grid_x(max((array_size_width+threads_x.x-1)/threads_x.x,1),1);
    kernel_set_random<<<grid_x,threads_x>>>(dev_states,array_size_width,clock_for_rand);

    dim3 threads(16,16);
    dim3 grid(max((array_size_width+threads.x-1)/threads.x,1),max((array_size_height+threads.y-1)/threads.y,1));  
    kernel_random<<<grid,threads>>>(dev_random_array,array_size_width,array_size_height,dev_states);

    //copy out the result
    cuda_status = cudaMemcpy(random_array,dev_random_array,sizeof(float)*array_size_width*array_size_height,cudaMemcpyDeviceToHost);//dev_depthMap
    if(cuda_status != cudaSuccess)
    {
        fprintf(stderr,"cudaMemcpy Failed");
        exit( EXIT_FAILURE );
    }

    for(int i=0;i<array_size_width*array_size_height;i++)
    {
        printf("%f\n",random_array[i]);
    }

    //free
    cudaFree(dev_random_array);
    cudaFree(dev_states);
    return 0;
}