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


__device__ bool equal(double *a, double *b, double err, struct dims dims) {
    for (size_t i = 0; i < dims.width * dims.height; i++) {
        if (std::abs(a[i] - b[i]) > err) {
            return false;
        }
    }
    return true;
}

__global__ void kernel(const double *sources, double *output, const double *buffer, struct dims dims, double err) {
    for (size_t y = 0; y < dims.width; y++) {
        for (size_t x = 0; x < dims.width; x++) {
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
            output[y * dims.width + x] = (
                    buffer[((y + 1) % dims.height) * dims.width + ((x + 0) % dims.width)] * 0.125 +
                    buffer[((y - 1 + dims.width) % dims.height) * dims.width + ((x + 0) % dims.width)] * 0.125 +
                    buffer[((y + 0) % dims.height) * dims.width + ((x + 1) % dims.width)] * 0.125 +
                    buffer[((y + 0) % dims.height) * dims.width + ((x - 1 + dims.width) % dims.width)] * 0.125 +
                    buffer[y * dims.width + x] * 0.5
            );

            if (sources[y * dims.width + x] != 0)
                output[y * dims.width + x] = sources[y * dims.width + x];
        }
    }
}

bool equal(std::vector<double> &a, std::vector<double> &b, double err) {
    for (size_t i = 0; i < a.size(); i++) {
        if (std::abs(a[i] - b[i]) > err) {
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
    size_t input_size = sources.size() * sizeof(double);

    /*
     * Allocate and copy memory to device.
     */
    gpuErrchk(cudaMalloc((void **) &dev_sources, input_size));
    gpuErrchk(cudaMalloc((void **) &dev_output, input_size));
    gpuErrchk(cudaMalloc((void **) &dev_buffer, input_size));
    gpuErrchk(cudaMemcpy((void *) dev_sources, (void *) sources.data(), input_size, cudaMemcpyHostToDevice));

    /*
     * Iterate computation
     */
    size_t iter;
    std::cout << "Processing dissipation." << std::endl;
    for (iter = 0; iter < max_iters; iter++) {
        /*
         * Launch our kernel
         */
        kernel<<<1, 1>>>(dev_sources, dev_output, dev_buffer, dims, err);
        gpuErrchk(cudaDeviceSynchronize());

        cudaMemcpyAsync((void *) output.data(), (void *) dev_output, input_size, cudaMemcpyDeviceToHost);
        cudaMemcpyAsync((void *) buffer.data(), (void *) dev_buffer, input_size, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        /*
         * Swap bufferss
         */
        double *temp = dev_output;
        dev_output = dev_buffer;
        dev_buffer = temp;

        if (equal(output, buffer, err)) {
            break;
        }
    }
    std::cout << "Stopped on iteration " << iter << std::endl;

    /*
     * Copy our data back to host.
     */
    cudaMemcpy((void *) output.data(), (void *) dev_output, input_size, cudaMemcpyDeviceToHost);

    return output;
}
