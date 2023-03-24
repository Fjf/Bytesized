#include <vector>
#include <iostream>
#include "kernels.cuh"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__device__
int getGlobalIdx_2D_2D() {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y)
                   + (threadIdx.y * blockDim.x) + threadIdx.x;
    return threadId;
}

__global__ void equal_kernel(double *a, double *b, struct dims dims, double err, bool *is_equal) {
    size_t i = getGlobalIdx_2D_2D();
    if (i >= dims.width * dims.height) return;

    if (std::fabs(a[i] - b[i]) > err) {
        *is_equal = false;
    }
}

__global__ void kernel(const double *sources, double *output, const double *buffer, struct dims dims, double err) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dims.width || y >= dims.height) return;

    /*
     * Take the four direct neighbours and set current value to sum of
     *
     *        0.125
     *  0.125 0.500 0.125
     *        0.125
     *
     * Edges wrap around.
     *
     * The sources are an infinite (fixed) energy point, reset them to their initial values after.
     */


    if (sources[y * dims.width + x] != 0) {
        output[y * dims.width + x] = sources[y * dims.width + x];
    } else {
        uint ny = (y - 1 + dims.height) % dims.height;
        uint py = (y + 1 + dims.height) % dims.height;
        uint nx = (x - 1 + dims.width) % dims.width;
        uint px = (x + 1 + dims.width) % dims.width;

        output[y * dims.width + x] = (
                buffer[ny * dims.width + x] * 0.125 +
                buffer[py * dims.width + x] * 0.125 +
                buffer[y * dims.width + nx] * 0.125 +
                buffer[y * dims.width + px] * 0.125 +
                buffer[y * dims.width + x] * 0.5
        );
    }

}

bool equal(std::vector<double> &a, std::vector<double> &b, double err) {
    for (size_t i = 0; i < a.size(); i++) {
        if (std::fabs(a[i] - b[i]) > err) {
            return false;
        }
    }
    return true;
}

std::vector<double>
run_kernel(const std::vector<double> &sources, const struct dims dims, const size_t max_iters, const double err) {
    // Initialize vector for our output array.
    std::vector<double> output(sources.size());
    std::vector<double> buffer(sources.size());
    /*
     * Create device pointer and calculate buffer size.
     */
    double *dev_sources;
    double *dev_buffer;
    double *dev_output;
    bool *dev_is_equal;
    size_t input_size = sources.size() * sizeof(double);

    /*
     * Allocate and copy memory to device.
     */
    gpuErrchk(cudaMalloc((void **) &dev_sources, input_size));
    gpuErrchk(cudaMalloc((void **) &dev_output, input_size));
    gpuErrchk(cudaMalloc((void **) &dev_buffer, input_size));
    gpuErrchk(cudaMalloc((void **) &dev_is_equal, sizeof(bool)));

    gpuErrchk(cudaMemcpy((void *) dev_sources, (void *) sources.data(), input_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy((void *) dev_buffer, (void *) sources.data(), input_size, cudaMemcpyHostToDevice));

    /*
     * Iterate computation
     */
    size_t iter;
    bool is_equal;
    std::cout << "Processing dissipation." << std::endl;
    for (iter = 0; iter < max_iters; iter++) {
        /*
         * Launch our kernel
         */
        uint block_size = 32;
        dim3 block = {block_size, block_size};
        dim3 grid = {
                uint((dims.width + block.x) / block.x),
                uint((dims.height + block.y) / block.y)
        };
        kernel<<<grid, block>>>(dev_sources, dev_output, dev_buffer, dims, err);
        gpuErrchk(cudaDeviceSynchronize());

        cudaMemset((void *) dev_is_equal, 1, 1);
        equal_kernel<<<grid, block>>>(dev_output, dev_buffer, dims, err, dev_is_equal);
        gpuErrchk(cudaDeviceSynchronize());
        cudaMemcpy((void *) &is_equal, (void *) dev_is_equal, sizeof(bool), cudaMemcpyDeviceToHost);

        /*
         * Swap buffers
         */
        double *temp = dev_output;
        dev_output = dev_buffer;
        dev_buffer = temp;

        if (is_equal)
            break;
    }
    std::cout << "Stopped on iteration " << iter << std::endl;

    /*
     * Copy our data back to host.
     */
    cudaMemcpy((void *) output.data(), (void *) dev_buffer, input_size, cudaMemcpyDeviceToHost);

    return output;
}
