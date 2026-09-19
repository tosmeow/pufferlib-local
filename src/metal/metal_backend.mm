#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "metal_backend.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <limits>
#include <string>

struct PufMetalContext {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* pipelines;
    NSMutableDictionary<NSString*, MPSMatrixMultiplication*>* gemm_kernels;
    std::string device_name;
    std::string last_error;
};

struct PufMetalBuffer {
    id<MTLBuffer> buffer;
    size_t logical_size;
};

struct PufMetalCommandBatch {
    PufMetalContext* context;
    id<MTLCommandBuffer> command_buffer;
    bool committed;
    bool completed;
};

struct ClampParams {
    float lower;
    float upper;
    uint32_t n;
};

struct Transpose102Params {
    uint32_t a;
    uint32_t b;
    uint32_t c;
    uint32_t total;
};

struct MinGRUGateParams {
    uint32_t batch_size;
    uint32_t hidden_size;
    uint32_t total;
};

struct MinGRUScanParams {
    uint32_t batch_size;
    uint32_t horizon;
    uint32_t hidden_size;
    uint32_t total;
};

static void copy_error(char* dst, size_t capacity, const std::string& error) {
    if (dst == nullptr || capacity == 0) return;
    size_t length = std::min(capacity - 1, error.size());
    memcpy(dst, error.data(), length);
    dst[length] = '\0';
}

static int fail(PufMetalContext* context, PufMetalStatus status,
        const std::string& error) {
    if (context != nullptr) context->last_error = error;
    return status;
}

static std::string ns_error(NSError* error) {
    if (error == nil) return "unknown Metal error";
    return std::string(error.localizedDescription.UTF8String ?: "unknown Metal error");
}

static bool has_bytes(const PufMetalBuffer* buffer, size_t elem_size, uint64_t n) {
    if (buffer == nullptr || elem_size == 0) return false;
    return n <= buffer->logical_size / elem_size;
}

static NSString* gemm_key(uint32_t m, uint32_t n, uint32_t k,
        bool transpose_lhs, bool transpose_rhs, float alpha, float beta) {
    // Use the exact scalar bit patterns: kernels with distinct alpha/beta
    // values must not share an MPSMatrixMultiplication instance.
    uint32_t alpha_bits;
    uint32_t beta_bits;
    memcpy(&alpha_bits, &alpha, sizeof(alpha_bits));
    memcpy(&beta_bits, &beta, sizeof(beta_bits));
    return [NSString stringWithFormat:@"%u:%u:%u:%d:%d:%08x:%08x",
        m, n, k, transpose_lhs, transpose_rhs, alpha_bits, beta_bits];
}

static MPSMatrixMultiplication* get_gemm_kernel(PufMetalContext* context,
        uint32_t m, uint32_t n, uint32_t k, bool transpose_lhs,
        bool transpose_rhs, float alpha, float beta) {
    NSString* key = gemm_key(
        m, n, k, transpose_lhs, transpose_rhs, alpha, beta);
    MPSMatrixMultiplication* cached = context->gemm_kernels[key];
    if (cached != nil) return cached;

    MPSMatrixMultiplication* kernel = [[MPSMatrixMultiplication alloc]
        initWithDevice:context->device
        transposeLeft:transpose_lhs
        transposeRight:transpose_rhs
        resultRows:m
        resultColumns:n
        interiorColumns:k
        alpha:alpha
        beta:beta];
    if (kernel == nil) {
        context->last_error = "Failed to create MPS GEMM kernel";
        return nil;
    }
    context->gemm_kernels[key] = kernel;
    return kernel;
}

static id<MTLComputePipelineState> get_pipeline(
        PufMetalContext* context, const char* function_name) {
    NSString* key = [NSString stringWithUTF8String:function_name];
    id<MTLComputePipelineState> cached = context->pipelines[key];
    if (cached != nil) return cached;

    id<MTLFunction> function = [context->library newFunctionWithName:key];
    if (function == nil) {
        context->last_error = "Metal function not found: " + std::string(function_name);
        return nil;
    }

    NSError* error = nil;
    id<MTLComputePipelineState> pipeline =
        [context->device newComputePipelineStateWithFunction:function error:&error];
    if (pipeline == nil) {
        context->last_error = "Failed to build pipeline "
            + std::string(function_name) + ": " + ns_error(error);
        return nil;
    }
    context->pipelines[key] = pipeline;
    return pipeline;
}

template <typename Bind>
static int encode_dispatch(PufMetalContext* context,
        id<MTLCommandBuffer> command_buffer, const char* function_name,
        uint32_t threads, Bind bind) {
    if (threads == 0) return PUF_METAL_OK;
    id<MTLComputePipelineState> pipeline = get_pipeline(context, function_name);
    if (pipeline == nil) return PUF_METAL_PIPELINE_ERROR;

    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
        return fail(context, PUF_METAL_EXECUTION_ERROR,
            "Failed to create Metal compute encoder");
    }
    [encoder setComputePipelineState:pipeline];
    bind(encoder);

    NSUInteger execution_width = pipeline.threadExecutionWidth;
    NSUInteger group_size = std::min<NSUInteger>(
        pipeline.maxTotalThreadsPerThreadgroup, 256);
    group_size = std::max(execution_width,
        (group_size / execution_width) * execution_width);
    [encoder dispatchThreads:MTLSizeMake(threads, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(group_size, 1, 1)];
    [encoder endEncoding];
    return PUF_METAL_OK;
}

template <typename Bind>
static int dispatch(PufMetalContext* context, const char* function_name,
        uint32_t threads, Bind bind) {
    @autoreleasepool {
        if (context == nullptr) return PUF_METAL_INVALID_ARGUMENT;
        context->last_error.clear();
        if (threads == 0) return PUF_METAL_OK;

        id<MTLCommandBuffer> command_buffer = [context->queue commandBuffer];
        if (command_buffer == nil) {
            return fail(context, PUF_METAL_EXECUTION_ERROR,
                "Failed to create Metal command buffer");
        }
        int status = encode_dispatch(
            context, command_buffer, function_name, threads, bind);
        if (status != PUF_METAL_OK) return status;

        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status == MTLCommandBufferStatusError) {
            return fail(context, PUF_METAL_EXECUTION_ERROR,
                "Metal command failed: " + ns_error(command_buffer.error));
        }
        return PUF_METAL_OK;
    }
}

extern "C" int puf_metal_is_available(void) {
    @autoreleasepool {
        return MTLCreateSystemDefaultDevice() != nil;
    }
}

extern "C" PufMetalContext* puf_metal_create(
        const char* library_path, char* error_out, size_t error_capacity) {
    @autoreleasepool {
        if (library_path == nullptr || library_path[0] == '\0') {
            copy_error(error_out, error_capacity, "Metal library path is empty");
            return nullptr;
        }

        auto* context = new PufMetalContext();
        context->device = MTLCreateSystemDefaultDevice();
        if (context->device == nil) {
            copy_error(error_out, error_capacity, "No Metal device is available");
            delete context;
            return nullptr;
        }
        context->queue = [context->device newCommandQueue];
        context->pipelines = [NSMutableDictionary dictionary];
        context->gemm_kernels = [NSMutableDictionary dictionary];
        context->device_name = context->device.name.UTF8String ?: "Unknown Metal device";

        NSString* path = [NSString stringWithUTF8String:library_path];
        NSError* error = nil;
        if ([path.pathExtension.lowercaseString isEqualToString:@"metallib"]) {
            NSURL* url = [NSURL fileURLWithPath:path];
            context->library = [context->device newLibraryWithURL:url error:&error];
        } else {
            NSString* source = [NSString stringWithContentsOfFile:path
                encoding:NSUTF8StringEncoding error:&error];
            if (source == nil) {
                copy_error(error_out, error_capacity,
                    "Failed to read Metal source: " + ns_error(error));
                delete context;
                return nullptr;
            }
            MTLCompileOptions* options = [MTLCompileOptions new];
            if (@available(macOS 15.0, *)) {
                options.mathMode = MTLMathModeSafe;
            } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
                options.fastMathEnabled = NO;
#pragma clang diagnostic pop
            }
            context->library = [context->device newLibraryWithSource:source
                options:options error:&error];
        }

        if (context->library == nil) {
            copy_error(error_out, error_capacity,
                "Failed to load Metal library: " + ns_error(error));
            delete context;
            return nullptr;
        }
        if (context->queue == nil) {
            copy_error(error_out, error_capacity, "Failed to create Metal command queue");
            delete context;
            return nullptr;
        }
        copy_error(error_out, error_capacity, "");
        return context;
    }
}

extern "C" void puf_metal_destroy(PufMetalContext* context) {
    delete context;
}

extern "C" const char* puf_metal_device_name(const PufMetalContext* context) {
    return context == nullptr ? "" : context->device_name.c_str();
}

extern "C" const char* puf_metal_last_error(const PufMetalContext* context) {
    return context == nullptr ? "Invalid Metal context" : context->last_error.c_str();
}

extern "C" PufMetalBuffer* puf_metal_buffer_create_shared(
        PufMetalContext* context, size_t size_bytes) {
    @autoreleasepool {
        if (context == nullptr) return nullptr;
        context->last_error.clear();
        // Metal does not permit zero-length buffers. Preserve the requested
        // logical size while allocating one inert byte for empty tensors.
        size_t allocation_size = std::max<size_t>(size_bytes, 1);
        id<MTLBuffer> buffer = [context->device newBufferWithLength:allocation_size
            options:MTLResourceStorageModeShared];
        if (buffer == nil) {
            context->last_error = "Failed to allocate shared Metal buffer";
            return nullptr;
        }
        auto* result = new PufMetalBuffer();
        result->buffer = buffer;
        result->logical_size = size_bytes;
        return result;
    }
}

extern "C" void puf_metal_buffer_destroy(PufMetalBuffer* buffer) {
    delete buffer;
}

extern "C" void* puf_metal_buffer_contents(PufMetalBuffer* buffer) {
    return buffer == nullptr ? nullptr : buffer->buffer.contents;
}

extern "C" size_t puf_metal_buffer_size(const PufMetalBuffer* buffer) {
    return buffer == nullptr ? 0 : buffer->logical_size;
}

extern "C" int puf_metal_fill_f32(PufMetalContext* context,
        PufMetalBuffer* dst, float value, uint32_t n) {
    if (!has_bytes(dst, sizeof(float), n)) {
        return fail(context, PUF_METAL_INVALID_ARGUMENT,
            "fill_f32 destination is smaller than n floats");
    }
    return dispatch(context, "fill_f32", n,
        [=](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:dst->buffer offset:0 atIndex:0];
            [encoder setBytes:&value length:sizeof(value) atIndex:1];
            [encoder setBytes:&n length:sizeof(n) atIndex:2];
        });
}

extern "C" int puf_metal_add_f32(PufMetalContext* context,
        PufMetalBuffer* dst, const PufMetalBuffer* src, uint32_t n) {
    if (!has_bytes(dst, sizeof(float), n) || !has_bytes(src, sizeof(float), n)) {
        return fail(context, PUF_METAL_INVALID_ARGUMENT,
            "add_f32 buffer is smaller than n floats");
    }
    return dispatch(context, "add_f32", n,
        [=](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:dst->buffer offset:0 atIndex:0];
            [encoder setBuffer:src->buffer offset:0 atIndex:1];
            [encoder setBytes:&n length:sizeof(n) atIndex:2];
        });
}

extern "C" int puf_metal_clamp_f32(PufMetalContext* context,
        PufMetalBuffer* dst, float lower, float upper, uint32_t n) {
    if (!has_bytes(dst, sizeof(float), n)) {
        return fail(context, PUF_METAL_INVALID_ARGUMENT,
            "clamp_f32 destination is smaller than n floats");
    }
    ClampParams params = {.lower = lower, .upper = upper, .n = n};
    return dispatch(context, "clamp_f32", n,
        [=](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:dst->buffer offset:0 atIndex:0];
            [encoder setBytes:&params length:sizeof(params) atIndex:1];
        });
}

extern "C" int puf_metal_cast_u8_f32(PufMetalContext* context,
        PufMetalBuffer* dst, const PufMetalBuffer* src, uint32_t n) {
    if (!has_bytes(dst, sizeof(float), n) || !has_bytes(src, sizeof(uint8_t), n)) {
        return fail(context, PUF_METAL_INVALID_ARGUMENT,
            "cast_u8_f32 buffer is smaller than n elements");
    }
    return dispatch(context, "cast_u8_f32", n,
        [=](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:dst->buffer offset:0 atIndex:0];
            [encoder setBuffer:src->buffer offset:0 atIndex:1];
            [encoder setBytes:&n length:sizeof(n) atIndex:2];
        });
}

extern "C" int puf_metal_transpose_102_f32(PufMetalContext* context,
        PufMetalBuffer* dst, const PufMetalBuffer* src,
        uint32_t a, uint32_t b, uint32_t c) {
    uint64_t total64 = static_cast<uint64_t>(a) * b * c;
    if (total64 > std::numeric_limits<uint32_t>::max()) {
        return fail(context, PUF_METAL_INVALID_ARGUMENT,
            "transpose_102_f32 tensor is too large");
    }
    if (dst == src && total64 > 1) {
        return fail(context, PUF_METAL_INVALID_ARGUMENT,
            "transpose_102_f32 does not support in-place operation");
    }
    if (!has_bytes(dst, sizeof(float), total64)
            || !has_bytes(src, sizeof(float), total64)) {
        return fail(context, PUF_METAL_INVALID_ARGUMENT,
            "transpose_102_f32 buffer is smaller than A*B*C floats");
    }
    Transpose102Params params = {
        .a = a, .b = b, .c = c, .total = static_cast<uint32_t>(total64)};
    return dispatch(context, "transpose_102_f32", params.total,
        [=](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:dst->buffer offset:0 atIndex:0];
            [encoder setBuffer:src->buffer offset:0 atIndex:1];
            [encoder setBytes:&params length:sizeof(params) atIndex:2];
        });
}

static int encode_gemm_f32(PufMetalContext* context,
        id<MTLCommandBuffer> command_buffer,
        PufMetalBuffer* dst, const PufMetalBuffer* lhs,
        const PufMetalBuffer* rhs, uint32_t m, uint32_t n, uint32_t k,
        int transpose_lhs, int transpose_rhs, float alpha, float beta) {
    if ((transpose_lhs != 0 && transpose_lhs != 1)
            || (transpose_rhs != 0 && transpose_rhs != 1)) {
        return fail(context, PUF_METAL_INVALID_ARGUMENT,
            "gemm_f32 transpose flags must be 0 or 1");
    }
    // No matrix element is read or written when either output dimension
    // is empty, matching the CUDA wrapper's effective behavior.
    if (m == 0 || n == 0) return PUF_METAL_OK;

    uint64_t lhs_elements = static_cast<uint64_t>(m) * k;
    uint64_t rhs_elements = static_cast<uint64_t>(k) * n;
    uint64_t dst_elements = static_cast<uint64_t>(m) * n;
    if (!has_bytes(lhs, sizeof(float), lhs_elements)
            || !has_bytes(rhs, sizeof(float), rhs_elements)
            || !has_bytes(dst, sizeof(float), dst_elements)) {
        return fail(context, PUF_METAL_INVALID_ARGUMENT,
            "gemm_f32 buffer is smaller than its matrix dimensions");
    }
    if (dst == lhs || dst == rhs || lhs == rhs) {
        return fail(context, PUF_METAL_INVALID_ARGUMENT,
            "gemm_f32 buffers may not alias");
    }
    // MPS does not accept an empty interior dimension. Preserve BLAS GEMM
    // semantics for K=0 without constructing empty MPSMatrix objects.
    if (k == 0) {
        if (beta == 1.0f) return PUF_METAL_OK;
        if (dst_elements > std::numeric_limits<uint32_t>::max()) {
            return fail(context, PUF_METAL_INVALID_ARGUMENT,
                "gemm_f32 output is too large for empty-K scaling");
        }
        uint32_t count = static_cast<uint32_t>(dst_elements);
        if (beta == 0.0f) {
            float zero = 0.0f;
            return encode_dispatch(context, command_buffer, "fill_f32", count,
                [=](id<MTLComputeCommandEncoder> encoder) {
                    [encoder setBuffer:dst->buffer offset:0 atIndex:0];
                    [encoder setBytes:&zero length:sizeof(zero) atIndex:1];
                    [encoder setBytes:&count length:sizeof(count) atIndex:2];
                });
        }
        return encode_dispatch(context, command_buffer, "scale_f32", count,
            [=](id<MTLComputeCommandEncoder> encoder) {
                [encoder setBuffer:dst->buffer offset:0 atIndex:0];
                [encoder setBytes:&beta length:sizeof(beta) atIndex:1];
                [encoder setBytes:&count length:sizeof(count) atIndex:2];
            });
    }

    NSUInteger lhs_rows = transpose_lhs ? k : m;
    NSUInteger lhs_columns = transpose_lhs ? m : k;
    NSUInteger rhs_rows = transpose_rhs ? n : k;
    NSUInteger rhs_columns = transpose_rhs ? k : n;
    MPSMatrixDescriptor* lhs_descriptor =
        [MPSMatrixDescriptor matrixDescriptorWithRows:lhs_rows
            columns:lhs_columns
            rowBytes:lhs_columns * sizeof(float)
            dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* rhs_descriptor =
        [MPSMatrixDescriptor matrixDescriptorWithRows:rhs_rows
            columns:rhs_columns
            rowBytes:rhs_columns * sizeof(float)
            dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* dst_descriptor =
        [MPSMatrixDescriptor matrixDescriptorWithRows:m
            columns:n
            rowBytes:static_cast<NSUInteger>(n) * sizeof(float)
            dataType:MPSDataTypeFloat32];
    MPSMatrix* lhs_matrix = [[MPSMatrix alloc]
        initWithBuffer:lhs->buffer descriptor:lhs_descriptor];
    MPSMatrix* rhs_matrix = [[MPSMatrix alloc]
        initWithBuffer:rhs->buffer descriptor:rhs_descriptor];
    MPSMatrix* dst_matrix = [[MPSMatrix alloc]
        initWithBuffer:dst->buffer descriptor:dst_descriptor];
    MPSMatrixMultiplication* kernel = get_gemm_kernel(context,
        m, n, k, transpose_lhs, transpose_rhs, alpha, beta);
    if (lhs_matrix == nil || rhs_matrix == nil || dst_matrix == nil
            || kernel == nil) {
        return fail(context, PUF_METAL_EXECUTION_ERROR,
            context->last_error.empty()
                ? "Failed to construct MPS GEMM matrices" : context->last_error);
    }

    [kernel encodeToCommandBuffer:command_buffer
        leftMatrix:lhs_matrix rightMatrix:rhs_matrix resultMatrix:dst_matrix];
    return PUF_METAL_OK;
}

extern "C" int puf_metal_gemm_f32(PufMetalContext* context,
        PufMetalBuffer* dst, const PufMetalBuffer* lhs,
        const PufMetalBuffer* rhs, uint32_t m, uint32_t n, uint32_t k,
        int transpose_lhs, int transpose_rhs, float alpha, float beta) {
    @autoreleasepool {
        if (context == nullptr) return PUF_METAL_INVALID_ARGUMENT;
        context->last_error.clear();
        id<MTLCommandBuffer> command_buffer = [context->queue commandBuffer];
        if (command_buffer == nil) {
            return fail(context, PUF_METAL_EXECUTION_ERROR,
                "Failed to create Metal command buffer for GEMM");
        }
        int status = encode_gemm_f32(context, command_buffer,
            dst, lhs, rhs, m, n, k, transpose_lhs, transpose_rhs, alpha, beta);
        if (status != PUF_METAL_OK) return status;
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status == MTLCommandBufferStatusError) {
            return fail(context, PUF_METAL_EXECUTION_ERROR,
                "Metal GEMM command failed: " + ns_error(command_buffer.error));
        }
        return PUF_METAL_OK;
    }
}

extern "C" PufMetalCommandBatch* puf_metal_command_batch_create(
        PufMetalContext* context) {
    @autoreleasepool {
        if (context == nullptr) return nullptr;
        context->last_error.clear();
        id<MTLCommandBuffer> command_buffer = [context->queue commandBuffer];
        if (command_buffer == nil) {
            context->last_error = "Failed to create Metal command batch";
            return nullptr;
        }
        auto* batch = new PufMetalCommandBatch();
        batch->context = context;
        batch->command_buffer = command_buffer;
        batch->committed = false;
        batch->completed = false;
        return batch;
    }
}

extern "C" void puf_metal_command_batch_destroy(PufMetalCommandBatch* batch) {
    @autoreleasepool {
        if (batch == nullptr) return;
        // Do not release a live command buffer while callers may also release
        // its shared buffers. Explicit wait remains the error-reporting path.
        if (batch->committed && !batch->completed) {
            [batch->command_buffer waitUntilCompleted];
        }
        delete batch;
    }
}

extern "C" int puf_metal_command_batch_gemm_f32(
        PufMetalCommandBatch* batch, PufMetalBuffer* dst,
        const PufMetalBuffer* lhs, const PufMetalBuffer* rhs,
        uint32_t m, uint32_t n, uint32_t k,
        int transpose_lhs, int transpose_rhs, float alpha, float beta) {
    @autoreleasepool {
        if (batch == nullptr || batch->context == nullptr) {
            return PUF_METAL_INVALID_ARGUMENT;
        }
        PufMetalContext* context = batch->context;
        context->last_error.clear();
        if (batch->committed) {
            return fail(context, PUF_METAL_INVALID_ARGUMENT,
                "Cannot encode into a committed Metal command batch");
        }
        return encode_gemm_f32(context, batch->command_buffer,
            dst, lhs, rhs, m, n, k,
            transpose_lhs, transpose_rhs, alpha, beta);
    }
}

extern "C" int puf_metal_command_batch_linear_forward_f32(
        PufMetalCommandBatch* batch, PufMetalBuffer* output,
        const PufMetalBuffer* input, const PufMetalBuffer* weight,
        uint32_t batch_size, uint32_t input_size, uint32_t output_size) {
    return puf_metal_command_batch_gemm_f32(batch,
        output, input, weight, batch_size, output_size, input_size,
        0, 1, 1.0f, 0.0f);
}

extern "C" int puf_metal_command_batch_linear_backward_f32(
        PufMetalCommandBatch* batch,
        PufMetalBuffer* grad_input, PufMetalBuffer* grad_weight,
        const PufMetalBuffer* grad_output, const PufMetalBuffer* input,
        const PufMetalBuffer* weight,
        uint32_t batch_size, uint32_t input_size, uint32_t output_size) {
    if (batch == nullptr || batch->context == nullptr) {
        return PUF_METAL_INVALID_ARGUMENT;
    }
    PufMetalContext* context = batch->context;
    context->last_error.clear();
    uint64_t input_elements = static_cast<uint64_t>(batch_size) * input_size;
    uint64_t output_elements = static_cast<uint64_t>(batch_size) * output_size;
    uint64_t weight_elements = static_cast<uint64_t>(output_size) * input_size;
    if (!has_bytes(grad_input, sizeof(float), input_elements)
            || !has_bytes(grad_weight, sizeof(float), weight_elements)
            || !has_bytes(grad_output, sizeof(float), output_elements)
            || !has_bytes(input, sizeof(float), input_elements)
            || !has_bytes(weight, sizeof(float), weight_elements)) {
        return fail(context, PUF_METAL_INVALID_ARGUMENT,
            "linear_backward_f32 buffer is smaller than its tensor dimensions");
    }
    if (grad_input == grad_weight || grad_input == grad_output
            || grad_input == input || grad_input == weight
            || grad_weight == grad_output || grad_weight == input
            || grad_weight == weight || grad_output == input
            || grad_output == weight || input == weight) {
        return fail(context, PUF_METAL_INVALID_ARGUMENT,
            "linear_backward_f32 buffers may not alias");
    }
    if (batch->committed) {
        return fail(context, PUF_METAL_INVALID_ARGUMENT,
            "Cannot encode into a committed Metal command batch");
    }

    // grad_weight[O,I] = grad_output[B,O]^T @ input[B,I]
    int status = encode_gemm_f32(context, batch->command_buffer,
        grad_weight, grad_output, input,
        output_size, input_size, batch_size,
        1, 0, 1.0f, 0.0f);
    if (status != PUF_METAL_OK) return status;
    // grad_input[B,I] = grad_output[B,O] @ weight[O,I]
    return encode_gemm_f32(context, batch->command_buffer,
        grad_input, grad_output, weight,
        batch_size, input_size, output_size,
        0, 0, 1.0f, 0.0f);
}

extern "C" int puf_metal_command_batch_mingru_gate_f32(
        PufMetalCommandBatch* batch,
        PufMetalBuffer* output, PufMetalBuffer* next_state,
        const PufMetalBuffer* combined, const PufMetalBuffer* state,
        const PufMetalBuffer* input,
        uint32_t batch_size, uint32_t hidden_size) {
    @autoreleasepool {
        if (batch == nullptr || batch->context == nullptr) {
            return PUF_METAL_INVALID_ARGUMENT;
        }
        PufMetalContext* context = batch->context;
        context->last_error.clear();
        if (batch->committed) {
            return fail(context, PUF_METAL_INVALID_ARGUMENT,
                "Cannot encode into a committed Metal command batch");
        }
        uint64_t total64 = static_cast<uint64_t>(batch_size) * hidden_size;
        uint64_t combined64 = 3 * total64;
        if (total64 > std::numeric_limits<uint32_t>::max()) {
            return fail(context, PUF_METAL_INVALID_ARGUMENT,
                "mingru_gate_f32 tensor is too large");
        }
        if (!has_bytes(output, sizeof(float), total64)
                || !has_bytes(next_state, sizeof(float), total64)
                || !has_bytes(combined, sizeof(float), combined64)
                || !has_bytes(state, sizeof(float), total64)
                || !has_bytes(input, sizeof(float), total64)) {
            return fail(context, PUF_METAL_INVALID_ARGUMENT,
                "mingru_gate_f32 buffer is smaller than its tensor dimensions");
        }
        if (output == next_state || output == combined || output == state
                || next_state == combined || next_state == state
                || next_state == input || combined == state
                || combined == input) {
            return fail(context, PUF_METAL_INVALID_ARGUMENT,
                "mingru_gate_f32 has an unsupported buffer alias");
        }
        MinGRUGateParams params = {
            .batch_size = batch_size,
            .hidden_size = hidden_size,
            .total = static_cast<uint32_t>(total64),
        };
        return encode_dispatch(context, batch->command_buffer,
            "mingru_gate_f32", params.total,
            [=](id<MTLComputeCommandEncoder> encoder) {
                [encoder setBuffer:output->buffer offset:0 atIndex:0];
                [encoder setBuffer:next_state->buffer offset:0 atIndex:1];
                [encoder setBuffer:combined->buffer offset:0 atIndex:2];
                [encoder setBuffer:state->buffer offset:0 atIndex:3];
                [encoder setBuffer:input->buffer offset:0 atIndex:4];
                [encoder setBytes:&params length:sizeof(params) atIndex:5];
            });
    }
}

extern "C" int puf_metal_command_batch_mingru_scan_forward_f32(
        PufMetalCommandBatch* batch,
        PufMetalBuffer* output, PufMetalBuffer* next_state,
        PufMetalBuffer* a_star, PufMetalBuffer* s_values,
        PufMetalBuffer* log_values,
        const PufMetalBuffer* combined, const PufMetalBuffer* state,
        const PufMetalBuffer* input,
        uint32_t batch_size, uint32_t horizon, uint32_t hidden_size) {
    @autoreleasepool {
        if (batch == nullptr || batch->context == nullptr) {
            return PUF_METAL_INVALID_ARGUMENT;
        }
        PufMetalContext* context = batch->context;
        context->last_error.clear();
        if (batch->committed) {
            return fail(context, PUF_METAL_INVALID_ARGUMENT,
                "Cannot encode into a committed Metal command batch");
        }
        if (horizon == std::numeric_limits<uint32_t>::max()) {
            return fail(context, PUF_METAL_INVALID_ARGUMENT,
                "mingru_scan_forward_f32 horizon is too large");
        }
        uint64_t state64 = static_cast<uint64_t>(batch_size) * hidden_size;
        uint64_t sequence64 = state64 * horizon;
        uint64_t combined64 = 3 * sequence64;
        uint64_t checkpoints64 = state64 * (static_cast<uint64_t>(horizon) + 1);
        if (state64 > std::numeric_limits<uint32_t>::max()) {
            return fail(context, PUF_METAL_INVALID_ARGUMENT,
                "mingru_scan_forward_f32 dispatch is too large");
        }
        if (!has_bytes(output, sizeof(float), sequence64)
                || !has_bytes(next_state, sizeof(float), state64)
                || !has_bytes(a_star, sizeof(float), checkpoints64)
                || !has_bytes(s_values, sizeof(float), checkpoints64)
                || !has_bytes(log_values, sizeof(float), checkpoints64)
                || !has_bytes(combined, sizeof(float), combined64)
                || !has_bytes(state, sizeof(float), state64)
                || !has_bytes(input, sizeof(float), sequence64)) {
            return fail(context, PUF_METAL_INVALID_ARGUMENT,
                "mingru_scan_forward_f32 buffer is smaller than its tensor dimensions");
        }
        const PufMetalBuffer* buffers[] = {
            output, next_state, a_star, s_values, log_values,
            combined, state, input,
        };
        for (size_t i = 0; i < sizeof(buffers) / sizeof(buffers[0]); ++i) {
            for (size_t j = i + 1; j < sizeof(buffers) / sizeof(buffers[0]); ++j) {
                // In-place highway output is safe because each thread reads
                // and writes only its own [B,T,H] lane.
                if ((i == 0 && j == 7) || buffers[i] != buffers[j]) continue;
                return fail(context, PUF_METAL_INVALID_ARGUMENT,
                    "mingru_scan_forward_f32 has an unsupported buffer alias");
            }
        }
        MinGRUScanParams params = {
            .batch_size = batch_size,
            .horizon = horizon,
            .hidden_size = hidden_size,
            .total = static_cast<uint32_t>(state64),
        };
        return encode_dispatch(context, batch->command_buffer,
            "mingru_scan_forward_f32", params.total,
            [=](id<MTLComputeCommandEncoder> encoder) {
                [encoder setBuffer:output->buffer offset:0 atIndex:0];
                [encoder setBuffer:next_state->buffer offset:0 atIndex:1];
                [encoder setBuffer:a_star->buffer offset:0 atIndex:2];
                [encoder setBuffer:s_values->buffer offset:0 atIndex:3];
                [encoder setBuffer:log_values->buffer offset:0 atIndex:4];
                [encoder setBuffer:combined->buffer offset:0 atIndex:5];
                [encoder setBuffer:state->buffer offset:0 atIndex:6];
                [encoder setBuffer:input->buffer offset:0 atIndex:7];
                [encoder setBytes:&params length:sizeof(params) atIndex:8];
            });
    }
}

extern "C" int puf_metal_command_batch_mingru_scan_backward_f32(
        PufMetalCommandBatch* batch,
        PufMetalBuffer* grad_combined, PufMetalBuffer* grad_state,
        PufMetalBuffer* grad_input,
        const PufMetalBuffer* combined,
        const PufMetalBuffer* a_star, const PufMetalBuffer* s_values,
        const PufMetalBuffer* input, const PufMetalBuffer* grad_output,
        const PufMetalBuffer* grad_next_state,
        uint32_t batch_size, uint32_t horizon, uint32_t hidden_size) {
    @autoreleasepool {
        if (batch == nullptr || batch->context == nullptr) {
            return PUF_METAL_INVALID_ARGUMENT;
        }
        PufMetalContext* context = batch->context;
        context->last_error.clear();
        if (batch->committed) {
            return fail(context, PUF_METAL_INVALID_ARGUMENT,
                "Cannot encode into a committed Metal command batch");
        }
        if (horizon == std::numeric_limits<uint32_t>::max()) {
            return fail(context, PUF_METAL_INVALID_ARGUMENT,
                "mingru_scan_backward_f32 horizon is too large");
        }
        uint64_t state64 = static_cast<uint64_t>(batch_size) * hidden_size;
        uint64_t sequence64 = state64 * horizon;
        uint64_t combined64 = 3 * sequence64;
        uint64_t checkpoints64 = state64 * (static_cast<uint64_t>(horizon) + 1);
        if (state64 > std::numeric_limits<uint32_t>::max()) {
            return fail(context, PUF_METAL_INVALID_ARGUMENT,
                "mingru_scan_backward_f32 dispatch is too large");
        }
        if (!has_bytes(grad_combined, sizeof(float), combined64)
                || !has_bytes(grad_state, sizeof(float), state64)
                || !has_bytes(grad_input, sizeof(float), sequence64)
                || !has_bytes(combined, sizeof(float), combined64)
                || !has_bytes(a_star, sizeof(float), checkpoints64)
                || !has_bytes(s_values, sizeof(float), checkpoints64)
                || !has_bytes(input, sizeof(float), sequence64)
                || !has_bytes(grad_output, sizeof(float), sequence64)
                || !has_bytes(grad_next_state, sizeof(float), state64)) {
            return fail(context, PUF_METAL_INVALID_ARGUMENT,
                "mingru_scan_backward_f32 buffer is smaller than its tensor dimensions");
        }
        const PufMetalBuffer* buffers[] = {
            grad_combined, grad_state, grad_input, combined, a_star,
            s_values, input, grad_output, grad_next_state,
        };
        for (size_t i = 0; i < sizeof(buffers) / sizeof(buffers[0]); ++i) {
            for (size_t j = i + 1; j < sizeof(buffers) / sizeof(buffers[0]); ++j) {
                if (buffers[i] == buffers[j]) {
                    return fail(context, PUF_METAL_INVALID_ARGUMENT,
                        "mingru_scan_backward_f32 buffers may not alias");
                }
            }
        }
        MinGRUScanParams params = {
            .batch_size = batch_size,
            .horizon = horizon,
            .hidden_size = hidden_size,
            .total = static_cast<uint32_t>(state64),
        };
        return encode_dispatch(context, batch->command_buffer,
            "mingru_scan_backward_f32", params.total,
            [=](id<MTLComputeCommandEncoder> encoder) {
                [encoder setBuffer:grad_combined->buffer offset:0 atIndex:0];
                [encoder setBuffer:grad_state->buffer offset:0 atIndex:1];
                [encoder setBuffer:grad_input->buffer offset:0 atIndex:2];
                [encoder setBuffer:combined->buffer offset:0 atIndex:3];
                [encoder setBuffer:a_star->buffer offset:0 atIndex:4];
                [encoder setBuffer:s_values->buffer offset:0 atIndex:5];
                [encoder setBuffer:input->buffer offset:0 atIndex:6];
                [encoder setBuffer:grad_output->buffer offset:0 atIndex:7];
                [encoder setBuffer:grad_next_state->buffer offset:0 atIndex:8];
                [encoder setBytes:&params length:sizeof(params) atIndex:9];
            });
    }
}

extern "C" int puf_metal_command_batch_commit(PufMetalCommandBatch* batch) {
    @autoreleasepool {
        if (batch == nullptr || batch->context == nullptr) {
            return PUF_METAL_INVALID_ARGUMENT;
        }
        PufMetalContext* context = batch->context;
        context->last_error.clear();
        if (batch->committed) {
            return fail(context, PUF_METAL_INVALID_ARGUMENT,
                "Metal command batch was already committed");
        }
        [batch->command_buffer commit];
        batch->committed = true;
        return PUF_METAL_OK;
    }
}

extern "C" int puf_metal_command_batch_wait(PufMetalCommandBatch* batch) {
    @autoreleasepool {
        if (batch == nullptr || batch->context == nullptr) {
            return PUF_METAL_INVALID_ARGUMENT;
        }
        PufMetalContext* context = batch->context;
        context->last_error.clear();
        if (!batch->committed) {
            return fail(context, PUF_METAL_INVALID_ARGUMENT,
                "Metal command batch must be committed before waiting");
        }
        if (!batch->completed) {
            [batch->command_buffer waitUntilCompleted];
            batch->completed = true;
        }
        if (batch->command_buffer.status == MTLCommandBufferStatusError) {
            return fail(context, PUF_METAL_EXECUTION_ERROR,
                "Metal command batch failed: "
                    + ns_error(batch->command_buffer.error));
        }
        return PUF_METAL_OK;
    }
}
