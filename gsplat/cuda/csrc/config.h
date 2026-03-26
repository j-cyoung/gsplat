#ifndef GSPLAT_ENABLE_PREFILTER
#define GSPLAT_ENABLE_PREFILTER 1
#endif

#ifndef GSPLAT_ENABLE_SSAA
#define GSPLAT_ENABLE_SSAA 1
#endif

#define BLOCK_X 16
#define BLOCK_Y 16
#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define N_THREADS 256

#define MAX_REGISTER_CHANNELS 3

#define CUDA_CALL(x)                                                           \
    do {                                                                       \
        if ((x) != cudaSuccess) {                                              \
            printf(                                                            \
                "Error at %s:%d - %s\n",                                       \
                __FILE__,                                                      \
                __LINE__,                                                      \
                cudaGetErrorString(cudaGetLastError())                         \
            );                                                                 \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)
