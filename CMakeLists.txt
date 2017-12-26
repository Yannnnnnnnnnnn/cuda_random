CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(random)

find_package(CUDA QUIET REQUIRED)

MESSAGE(${CUDA_curand_LIBRARY})

# Specify binary name and source file to build it from
CUDA_ADD_EXECUTABLE(random random.cu)
CUDA_ADD_EXECUTABLE(random_gen random_gen.cu)

CUDA_ADD_EXECUTABLE(host_random host_random.cu)
target_link_libraries(host_random ${CUDA_curand_LIBRARY})
