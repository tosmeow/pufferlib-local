#include "gfr.h"

#define OBS_SIZE GFR_OBS_SIZE
#define NUM_ATNS 1
#define ACT_SIZES {GFR_DEFAULT_ACTIONS}
#define OBS_TENSOR_T FloatTensor

#define Env GFR
#define Log GFRLog
#include "vecenv.h"

#ifdef PUFFERGF_CPU_STUB_CUDA
cudaError_t cudaHostAlloc(void** ptr, size_t size, unsigned int flags) {
    (void)flags;
    *ptr = calloc(1, size);
    return cudaSuccess;
}

cudaError_t cudaMalloc(void** ptr, size_t size) {
    *ptr = calloc(1, size);
    return cudaSuccess;
}

cudaError_t cudaMemcpy(void* dst, const void* src, size_t size, cudaMemcpyKind kind) {
    (void)kind;
    memcpy(dst, src, size);
    return cudaSuccess;
}

cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t size,
        cudaMemcpyKind kind, cudaStream_t stream) {
    (void)stream;
    return cudaMemcpy(dst, src, size, kind);
}

cudaError_t cudaMemset(void* ptr, int value, size_t size) {
    memset(ptr, value, size);
    return cudaSuccess;
}

cudaError_t cudaFree(void* ptr) {
    free(ptr);
    return cudaSuccess;
}

cudaError_t cudaFreeHost(void* ptr) {
    free(ptr);
    return cudaSuccess;
}

cudaError_t cudaSetDevice(int device) {
    (void)device;
    return cudaSuccess;
}

cudaError_t cudaDeviceSynchronize(void) {
    return cudaSuccess;
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    (void)stream;
    return cudaSuccess;
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t* stream, unsigned int flags) {
    (void)flags;
    *stream = NULL;
    return cudaSuccess;
}

cudaError_t cudaStreamQuery(cudaStream_t stream) {
    (void)stream;
    return cudaSuccess;
}

const char* cudaGetErrorString(cudaError_t error) {
    (void)error;
    return "PUFFERGF_CPU_STUB_CUDA";
}
#endif

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->scenario = GFR_ACADEMY_EMPTY_GOAL_CLOSE;
    DictItem* scenario = dict_get_unsafe(kwargs, "scenario");
    if (scenario != NULL) {
        env->scenario = (int)scenario->value;
    }
}

void my_log(GFRLog* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "goals_for", log->goals_for);
    dict_set(out, "goals_against", log->goals_against);
}
